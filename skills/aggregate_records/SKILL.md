# aggregate_records

Use this process-layer Skill for extraction tasks whose `capability_type` is
`aggregate`. It groups repeated source rows by explicit keys and emits named,
auditable aggregation columns. Do not use it for direct scalar extraction,
cross-table joins, or semantic text extraction.

## Input contract

`group_keys_json` is a JSON list of source columns. `aggregations_json` maps
each source column to a list of operations. Supported operations are `count`,
`mean`, `min`, `max`, `sum`, `first`, `last`, and `list`.

```json
{
  "group_keys_json": "[\"case_id\", \"item\"]",
  "aggregations_json": "{\"value\": [\"count\", \"mean\", \"list\"]}"
}
```
