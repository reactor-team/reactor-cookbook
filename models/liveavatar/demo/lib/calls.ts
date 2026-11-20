import "server-only";

import type { AvatarSession } from "./reactor";

/** The avatar session of each call this server started, by room. One server process; enough for a demo. */
export const calls = new Map<string, AvatarSession>();
