# Upstream attribution

This adapter imports History-Forcing at revision
`9d69c1e5da2beb219e635f4400b8cb494759ba06`; it does not copy or modify its model
implementation. The build fetches that source, and startup downloads weights.

- [History-Forcing root license](https://github.com/ziplab/History-Forcing/blob/9d69c1e5da2beb219e635f4400b8cb494759ba06/LICENSE): Apache-2.0.
- [Vendored MemRoPE license](https://github.com/ziplab/History-Forcing/blob/9d69c1e5da2beb219e635f4400b8cb494759ba06/dmd/long_video/third_party/memrope/LICENSE): Apache-2.0.
- [Vendored attention source](https://github.com/ziplab/History-Forcing/blob/9d69c1e5da2beb219e635f4400b8cb494759ba06/dmd/long_video/third_party/memrope/attention.py): header identifies CC-BY-NC-SA-4.0 and credits the Self Forcing and MemRoPE authors.
- [DMD checkpoint](https://huggingface.co/xzxxntxdy/History-Forcing/tree/8604aafe4fada0311c7f5a993e7e620473c8e7ac): the model card contains no license field at this revision.
- [Wan2.1-T2V-1.3B assets](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a): retain their upstream licensing and attribution.

The conflicting MemRoPE declarations and unspecified checkpoint license are
recorded without attempting to resolve or override the authors' terms.
