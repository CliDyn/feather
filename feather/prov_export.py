"""Export a figure's provenance as W3C PROV-JSON.

Feather keeps provenance internally as flat JSON events (what the sidecar,
dashboard, prompts and ``feather verify`` want).  This module maps one
figure's record onto the PROV data model for interoperability, e.g. with
ESMValTool provenance records:

- ``prov:Entity``   — each input source, the figure, its LLM analysis
- ``prov:Activity`` — the pipeline run, the diagnostic, each regrid decision
- ``prov:Agent``    — feather (software, at a commit) and the LLM

See https://www.w3.org/submissions/prov-json/.
"""

from __future__ import annotations

import hashlib
from typing import Any

NS = "feather"
PREFIXES = {
    NS: "https://github.com/CliDyn/feather/provenance#",
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def _id(kind: str, *parts: Any) -> str:
    raw = "/".join(str(p) for p in parts)
    if len(raw) > 80 or any(c in raw for c in " :"):
        raw = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"{NS}:{kind}/{raw}"


def _attrs(src: dict, keys: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for k in keys:
        v = src.get(k)
        if v is None or v == [] or v == {}:
            continue
        out[f"{NS}:{k}"] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


def figure_to_prov(info: dict[str, Any]) -> dict[str, Any]:
    """PROV-JSON document for one figure (as built by
    :func:`feather.provenance_cli.figure_provenance`)."""
    doc: dict[str, Any] = {
        "prefix": dict(PREFIXES),
        "entity": {}, "activity": {}, "agent": {},
        "used": {}, "wasGeneratedBy": {}, "wasAssociatedWith": {},
        "wasDerivedFrom": {}, "wasInformedBy": {}, "wasAttributedTo": {},
    }
    n = 0

    def rel(kind: str, body: dict) -> None:
        nonlocal n
        n += 1
        doc[kind][f"_:{kind}{n}"] = body

    run = info.get("run") or {}
    run_id = info.get("run_id") or run.get("run_id") or "unknown"
    figure = info["figure"]
    diag = figure.split("/", 1)[0]

    fig_id = _id("figure", figure)
    doc["entity"][fig_id] = {
        "prov:type": f"{NS}:Figure", "prov:label": figure,
        f"{NS}:sidecar": info.get("sidecar"),
        **({"prov:generatedAtTime": info["generated_at"]} if info.get("generated_at") else {}),
    }

    software = _id("software", "feather", run.get("git_commit") or "unknown")
    doc["agent"][software] = {
        "prov:type": "prov:SoftwareAgent", "prov:label": "feather",
        **_attrs(run, ("feather_version", "git_commit", "git_branch", "git_dirty")),
    }
    run_act = _id("run", run_id)
    doc["activity"][run_act] = {
        "prov:type": f"{NS}:PipelineRun", "prov:label": run_id,
        **({"prov:startTime": run["started"]} if run.get("started") else {}),
        **({"prov:endTime": run["finished"]} if run.get("finished") else {}),
        **_attrs(run, ("hostname", "slurm_job_id")),
        **({f"{NS}:argv": " ".join(run["argv"])} if run.get("argv") else {}),
    }
    rel("wasAssociatedWith", {"prov:activity": run_act, "prov:agent": software})

    diag_act = _id("diagnostic", run_id, diag)
    doc["activity"][diag_act] = {"prov:type": f"{NS}:Diagnostic", "prov:label": diag}
    rel("wasInformedBy", {"prov:informed": diag_act, "prov:informant": run_act})
    rel("wasGeneratedBy", {"prov:entity": fig_id, "prov:activity": diag_act})

    for e in info.get("events", []):
        step = e.get("step")
        if step == "read":
            src = e.get("root") or e.get("first") or e.get("store") or e.get("catalog_key")
            ent = _id("input", e.get("role"), src, e.get("variable") or e.get("obs_variable"))
            doc["entity"][ent] = {
                "prov:type": f"{NS}:{(e.get('role') or 'input').capitalize()}Input",
                "prov:label": str(src),
                **_attrs(e, ("backend", "name", "dataset", "variable", "obs_variable",
                             "declared_experiment", "variant", "n_files", "identity",
                             "identity_method", "n_times", "n_missing_months")),
                **({f"{NS}:time_coverage": "/".join(e["time_coverage"])}
                   if e.get("time_coverage") else {}),
            }
            rel("used", {"prov:activity": diag_act, "prov:entity": ent})
            rel("wasDerivedFrom", {"prov:generatedEntity": fig_id, "prov:usedEntity": ent})
        elif step in ("regrid_method", "regrid", "convert"):
            act = _id(step, run_id, diag, e.get("variable"), e.get("used") or e.get("op"),
                      e.get("n_src"))
            doc["activity"][act] = {
                "prov:type": f"{NS}:{step}",
                **_attrs(e, ("variable", "requested", "used", "fallback", "reason",
                             "n_src", "n_tgt", "points_merged", "op", "factor",
                             "offset", "role")),
            }
            rel("wasInformedBy", {"prov:informed": diag_act, "prov:informant": act})
        elif step == "benchmark":
            ent = _id("benchmark", run_id, e.get("benchmark"), e.get("variable"))
            doc["entity"][ent] = {
                "prov:type": f"{NS}:BenchmarkEnsemble",
                "prov:label": f"{e.get('benchmark')} {e.get('variable')}",
                f"{NS}:n_used": e.get("n_used", 0),
                f"{NS}:used": ", ".join(e.get("used", [])),
                f"{NS}:excluded": ", ".join(
                    f"{x['member']} ({x['reason']})" for x in e.get("excluded", [])),
            }
            rel("used", {"prov:activity": diag_act, "prov:entity": ent})
            rel("wasDerivedFrom", {"prov:generatedEntity": fig_id, "prov:usedEntity": ent})

    interp = info.get("interpretation")
    if interp:
        llm = _id("llm", interp.get("provider"), interp.get("model"))
        doc["agent"][llm] = {"prov:type": "prov:SoftwareAgent",
                             "prov:label": interp.get("model")}
        ana = _id("analysis", figure)
        doc["entity"][ana] = {
            "prov:type": f"{NS}:Interpretation",
            **({"prov:generatedAtTime": interp["generated_at"]} if interp.get("generated_at") else {}),
            **_attrs(interp, ("schema", "sidecar_sha256", "figure_sha256",
                              "finish_reason", "attempts")),
        }
        rel("wasDerivedFrom", {"prov:generatedEntity": ana, "prov:usedEntity": fig_id})
        rel("wasAttributedTo", {"prov:entity": ana, "prov:agent": llm})

    return {k: v for k, v in doc.items() if v}
