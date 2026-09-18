import { z } from "zod";

export const TokenRequestSchema = z.object({
  session_id: z.string(),
  identity: z.string(),
  name: z.string().nullable().default(null),
});
export type TokenRequest = z.infer<typeof TokenRequestSchema>;

export const TokenResponseSchema = z.object({
  token: z.string(),
  url: z.string(),
  room: z.string(),
});
export type TokenResponse = z.infer<typeof TokenResponseSchema>;

export const RoomMetadataSchema = z.object({
  session_id: z.string(),
  // Edge TTS voice for the selected persona (e.g. "en-US-JennyNeural").
  // Optional: absent in older tokens. Carried on PARTICIPANT metadata by the
  // token minter (LiveKit auto-creates rooms with empty room metadata), so
  // the worker reads participant metadata as a fallback — see worker.py.
  voice: z.string().optional(),
});
export type RoomMetadata = z.infer<typeof RoomMetadataSchema>;
