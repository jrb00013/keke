#!/usr/bin/env python3
"""Static consistency checks for the Kubernetes manifests in k8s/.

Catches the failure modes that only surface at `kubectl apply` time:

  * two documents with the same kind/namespace/name (the previous
    deployment.yaml vs keke-deployment.yaml collision, where the second apply
    fails on the immutable selector),
  * a Service whose selector does not match any Deployment's pod template,
    or matches more than one (every Service selecting all pods),
  * a workload env var pointing at a Secret/ConfigMap key that is not defined.

Run:  python scripts/validate_k8s.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

K8S_DIR = Path(__file__).resolve().parent.parent / "k8s"


def load_docs():
    docs = []
    for path in sorted(K8S_DIR.glob("*.yaml")):
        with path.open() as handle:
            for doc in yaml.safe_load_all(handle):
                if doc:
                    docs.append((path.name, doc))
    return docs


def identity(doc):
    meta = doc.get("metadata", {})
    return (
        doc.get("kind"),
        meta.get("namespace", "default"),
        meta.get("name"),
    )


def pod_template_labels(doc):
    if doc.get("kind") not in ("Deployment", "StatefulSet", "DaemonSet"):
        return None
    return doc["spec"]["template"]["metadata"].get("labels", {})


def main() -> int:
    if not K8S_DIR.is_dir():
        print(f"no {K8S_DIR} directory")
        return 0

    docs = load_docs()
    errors: list[str] = []

    seen: dict[tuple, str] = {}
    for fname, doc in docs:
        ident = identity(doc)
        if None in ident:
            errors.append(f"{fname}: document missing kind/namespace/name: {ident}")
            continue
        if ident in seen:
            errors.append(
                f"duplicate object {ident[0]}/{ident[1]}/{ident[2]} "
                f"in both {seen[ident]} and {fname}"
            )
        else:
            seen[ident] = fname

    workloads = [(f, d) for f, d in docs if pod_template_labels(d)]
    for fname, svc in ((f, d) for f, d in docs if d.get("kind") == "Service"):
        selector = svc.get("spec", {}).get("selector") or {}
        if not selector:
            continue
        matches = [
            f"{wf}:{w['metadata']['name']}"
            for wf, w in workloads
            if all(pod_template_labels(w).get(k) == v for k, v in selector.items())
        ]
        if len(matches) != 1:
            errors.append(
                f"{fname}: Service/{svc['metadata']['name']} selector {selector} "
                f"matches {len(matches)} workloads ({', '.join(matches) or 'none'})"
            )

    secret_keys: set[str] = set()
    configmap_keys: dict[str, set[str]] = {}
    for _fname, doc in docs:
        if doc.get("kind") == "Secret":
            secret_keys.update((doc.get("stringData") or {}).keys())
            secret_keys.update((doc.get("data") or {}).keys())
        if doc.get("kind") == "ConfigMap":
            configmap_keys.setdefault(doc["metadata"]["name"], set()).update(
                (doc.get("data") or {}).keys()
            )

    # The example secret is the template for the out-of-band keke-secrets object.
    example = K8S_DIR / "secrets.yaml.example"
    if example.exists():
        with example.open() as handle:
            doc = yaml.safe_load(handle)
        secret_keys.update((doc.get("stringData") or {}).keys())

    for fname, doc in docs:
        template = pod_template_labels(doc)
        if not template:
            continue
        for container in doc["spec"]["template"]["spec"].get("containers", []):
            for env in container.get("env", []):
                ref = env.get("valueFrom") or {}
                if "secretKeyRef" in ref:
                    key = ref["secretKeyRef"]["key"]
                    if key not in secret_keys:
                        errors.append(
                            f"{fname}: Deployment/{doc['metadata']['name']} env "
                            f"{env['name']} references undefined Secret key '{key}'"
                        )
                if "configMapKeyRef" in ref:
                    name = ref["configMapKeyRef"]["name"]
                    key = ref["configMapKeyRef"]["key"]
                    if key not in configmap_keys.get(name, set()):
                        errors.append(
                            f"{fname}: Deployment/{doc['metadata']['name']} env "
                            f"{env['name']} references undefined ConfigMap "
                            f"'{name}' key '{key}'"
                        )

    if errors:
        print("k8s validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"k8s validation passed ({len(docs)} documents across {len(list(K8S_DIR.glob('*.yaml')))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
