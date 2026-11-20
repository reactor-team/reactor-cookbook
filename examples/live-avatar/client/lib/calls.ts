import "server-only";

import type { AvatarSession } from "./reactor";

/** The avatar session of each call this server started, by room. One server process holds them all. */
export const calls = new Map<string, AvatarSession>();
