# OpenDreamer conditioning images

Upload any PNG in this directory through `set_conditioning_image`. Each image
is an unmodified 640x360 frame extracted from the public VPT recording used by
the built-in demos. The adapter adds OpenDreamer's four-pixel top and bottom
padding and constructs the neutral 16-frame action context at runtime.

| File | Source frame | Scene |
| --- | ---: | --- |
| `meadow-pig-frame-0175.png` | 175 | Open meadow and forest |
| `pumpkin-tree-frame-0300.png` | 300 | Pumpkin patch beside a tree |
| `crafting-table-frame-3000.png` | 3000 | Outdoor crafting table |
| `furnace-frame-3900.png` | 3900 | Furnace at the forest edge |

The source recording is
`cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4` from the
[public OpenAI VPT index](https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json).
Its SHA-256 digest is
`c3dfede32353f7c19a297284ba671382bfbe32ac654077f1d37fe5bd26a41cfe`.
