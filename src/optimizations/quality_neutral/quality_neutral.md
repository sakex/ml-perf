# Quality Neutral

Quality Neutral optimization techniques should almost always be used as they make inference much faster and cheaper without any degradation on the model's output. Unfortunately, these techniques require more work to implement than the [Quality Detrimental](../quality_detrimental/quality_detrimental.md) ones.

It is important to understand [KV caching](./kv_caching.md) before [disaggregated serving](./disagg.md) because the later builds on the former.
