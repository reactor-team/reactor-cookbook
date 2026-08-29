"use client";

import { LongliveV2MainVideoView } from "@reactor-models/longlive-v2";

// The whole right side of the screen — black rounded panel,
// `<LongliveV2MainVideoView>` is a pre-bound `<ReactorView track="main_video">`
// from the typed SDK. No refs, no `srcObject`, no autoplay tricks.
export function Video() {
  return (
    <div className="relative h-full min-h-[40vh] w-full overflow-hidden rounded-lg border border-zinc-800 bg-black lg:min-h-0">
      <LongliveV2MainVideoView
        className="h-full w-full"
        videoObjectFit="contain"
      />
    </div>
  );
}
