# Demo Knowledge Base Packs

The repository now includes an expanded enterprise demo corpus designed to force multi-step tool behavior:

- document resolution from partial names
- hybrid retrieval across long files
- clause extraction and side-by-side clause comparison
- requirements mining from SHALL/MUST/REQ language
- parallel multi-document synthesis for executive decisions

## Long-Form Demo Pack

Core showcase files:

- `06_master_services_agreement_v1.md`
- `07_master_services_agreement_v2.md`
- `08_data_processing_addendum_global.md`
- `09_ai_ops_control_standard.md`
- `10_incident_communications_playbook.md`
- `11_vendor_security_schedule.md`

Existing baseline pack remains available:

- product docs: `01_*` to `05_*`
- runbooks: `runbook_*`
- API references: `api_*`

## Scenario-to-Pack Mapping

Scenario definitions live in `data/demo/demo_scenarios.json` (v2 structured schema).

- `utility_memory_finance_bootstrap`
  - Primary docs: all indexed docs via `list_indexed_docs`
  - Focus: utility + memory reliability
- `rag_resolution_and_search_strategy`
  - Primary docs: release notes, integrations, security/privacy
  - Focus: resolve + search strategy switching
- `rag_clause_navigation_and_extraction`
  - Primary docs: `06_master_services_agreement_v1.md`
  - Focus: structure scan + exact clause extraction
- `rag_requirements_traceability`
  - Primary docs: `09_ai_ops_control_standard.md`, `11_vendor_security_schedule.md`
  - Focus: requirements extraction and filter refinement
- `rag_structural_diff_contract_versions`
  - Primary docs: `06_master_services_agreement_v1.md`, `07_master_services_agreement_v2.md`
  - Focus: structural diff and change categorization
- `rag_clause_compare_conflict_review`
  - Primary docs: `06_master_services_agreement_v1.md`, `07_master_services_agreement_v2.md`
  - Focus: clause-by-clause contradiction review
- `parallel_rag_multi_doc_risk_board`
  - Primary docs: `08_data_processing_addendum_global.md`, `09_ai_ops_control_standard.md`, `10_incident_communications_playbook.md`
  - Focus: fan-out/fan-in synthesis across domains
- `executive_due_diligence_grand_finale`
  - Primary docs: `06_master_services_agreement_v2.md`, `08_data_processing_addendum_global.md`, `11_vendor_security_schedule.md`
  - Focus: executive recommendation with evidence and budget math

## Demo Runner

```bash
python run.py demo --list-scenarios
python run.py demo --scenario utility_memory_finance_bootstrap --verify
python run.py demo --scenario parallel_rag_multi_doc_risk_board --force-agent
python run.py demo --scenario all --session-mode scenario --verify
```

Notes:

- `--session-mode scenario` creates a fresh conversation context per scenario.
- `--session-mode suite` reuses one session across the full run.
- `--verify` prints heuristic `PASS/WARN/FAIL` checks per turn.
