# Demo clip provenance

The demo source clips in this directory are AI-generated originals, created for
this example. They contain no third-party footage, no identifiable people, and
no trademarked or copyrighted characters.

- Generated with Replicate `bytedance/seedance-1-lite` (text-to-video), 2026-07-13.
- Owned by Reactor; free to use, modify, and redistribute as part of this example.

| File         | Subject                                    | Demo slot              |
| ------------ | ------------------------------------------ | ---------------------- |
| `ball.mp4`   | a red rubber ball rolling across a table   | swap object            |
| `dog.mp4`    | a golden retriever walking across a garden | replace with reference |
| `figure.mp4` | a wooden artist's mannequin waving         | drag to animate        |

The reference images in `../refs/` are likewise generated originals — a single
clean subject on a plain backdrop, generated with Replicate
`black-forest-labs/flux-1.1-pro` (2026-07-13) at landscape 16:9 (1344×768) to
match the clips and the model's landscape output. Keeping every asset on one
aspect avoids the portrait-reference-into-landscape-output distortion.
