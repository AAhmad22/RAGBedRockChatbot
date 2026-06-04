# Architecture

See the diagram in the [README](../README.md#architecture).

## Design decisions

- **Managed retrieval (Bedrock Knowledge Bases)** over a hand-rolled vector
  pipeline: keeps the focus on deployment, guardrails, and evaluation rather
  than reinventing chunking/embedding. The roadmap includes a custom-retrieval
  variant to demonstrate the internals.
- **Serverless (Lambda + API Gateway)** over always-on compute: cheaper for a
  portfolio project; scales to zero.
- **OIDC for CI deploys** over static keys: no long-lived credentials in GitHub.

<!-- TODO: expand with tradeoffs, cost notes, and a sequence diagram. -->
