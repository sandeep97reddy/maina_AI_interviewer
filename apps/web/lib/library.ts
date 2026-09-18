"use client";

/**
 * Personal setup library — saved CVs, job descriptions, and companies in
 * `localStorage` (this browser only, no account, no backend).
 *
 * SSR safety (load-bearing): Next.js prerenders `"use client"` components on
 * the server where `localStorage` does not exist. Every read/write below is
 * guarded by `typeof localStorage === "undefined"` and callers must load in a
 * `useEffect` (empty initial state on first paint), mirroring `useMessages`.
 */

export interface SavedCv {
  id: string;
  /** First line of the CV text, truncated — shown on the card. */
  label: string;
  text: string;
  updatedAt: number;
}

export interface SavedJd {
  id: string;
  /** "Role · Company" style short label, truncated. */
  label: string;
  company: string;
  text: string;
  updatedAt: number;
}

export interface SavedCompany {
  id: string;
  name: string;
  updatedAt: number;
}

export interface Library {
  cvs: SavedCv[];
  jds: SavedJd[];
  companies: SavedCompany[];
}

export const EMPTY_LIBRARY: Library = { cvs: [], jds: [], companies: [] };

// Caps keep us far under the ~5MB localStorage quota.
const MAX_CVS = 5;
const MAX_JDS = 10;
const MAX_COMPANIES = 10;
const MAX_CHARS_PER_ITEM = 50_000;
const MAX_LABEL_CHARS = 60;

const KEYS = {
  cvs: "di.lib.cvs.v1",
  jds: "di.lib.jds.v1",
  companies: "di.lib.companies.v1",
} as const;

function store(): Storage | null {
  return typeof localStorage === "undefined" ? null : localStorage;
}

function newId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;
}

/** FNV-1a 32-bit hash — sync dedupe key (crypto.subtle is async; overkill). */
function hash(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

function firstLine(text: string): string {
  const line = text
    .split("\n")
    .map((l) => l.trim())
    .find((l) => l.length > 0);
  const raw = line && line.length > 0 ? line : text.trim().slice(0, 40);
  return raw.length > MAX_LABEL_CHARS
    ? `${raw.slice(0, MAX_LABEL_CHARS - 1)}…`
    : raw;
}

function read<T>(key: string): T[] {
  const s = store();
  if (!s) return [];
  try {
    const raw = s.getItem(key);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as T[]) : [];
  } catch {
    // Corrupt storage must never crash the page — start empty.
    return [];
  }
}

function write(key: string, value: unknown): void {
  const s = store();
  if (!s) return;
  try {
    s.setItem(key, JSON.stringify(value));
  } catch {
    // Quota exceeded or blocked storage — library just doesn't persist.
  }
}

/** Load the whole library. Safe to call during render (returns [] on server). */
export function loadLibrary(): Library {
  return {
    cvs: read<SavedCv>(KEYS.cvs),
    jds: read<SavedJd>(KEYS.jds),
    companies: read<SavedCompany>(KEYS.companies),
  };
}

/**
 * Upsert a CV by content hash (re-saving the same text bumps it to the top
 * instead of duplicating). Returns the new CV list, newest first.
 */
export function saveCv(text: string): SavedCv[] {
  const clean = text.trim().slice(0, MAX_CHARS_PER_ITEM);
  if (!clean) return read<SavedCv>(KEYS.cvs);
  const key = hash(clean);
  const rest = read<SavedCv>(KEYS.cvs).filter((c) => hash(c.text) !== key);
  const next: SavedCv[] = [
    {
      id: newId(),
      label: firstLine(clean),
      text: clean,
      updatedAt: Date.now(),
    },
    ...rest,
  ].slice(0, MAX_CVS);
  write(KEYS.cvs, next);
  return next;
}

/** Upsert a JD by content hash. Company snapshot is stored with the JD. */
export function saveJd(text: string, company: string): SavedJd[] {
  const clean = text.trim().slice(0, MAX_CHARS_PER_ITEM);
  if (!clean) return read<SavedJd>(KEYS.jds);
  const co = company.trim().slice(0, 120);
  const key = hash(`${clean}\n${co}`);
  const rest = read<SavedJd>(KEYS.jds).filter(
    (j) => hash(`${j.text}\n${j.company}`) !== key,
  );
  const labelBase = co ? `${firstLine(clean)} · ${co}` : firstLine(clean);
  const next: SavedJd[] = [
    {
      id: newId(),
      label:
        labelBase.length > MAX_LABEL_CHARS
          ? `${labelBase.slice(0, MAX_LABEL_CHARS - 1)}…`
          : labelBase,
      company: co,
      text: clean,
      updatedAt: Date.now(),
    },
    ...rest,
  ].slice(0, MAX_JDS);
  write(KEYS.jds, next);
  return next;
}

/** Upsert a company by case-insensitive name. Returns the new list. */
export function saveCompany(name: string): SavedCompany[] {
  const clean = name.trim().slice(0, 120);
  if (!clean) return read<SavedCompany>(KEYS.companies);
  const rest = read<SavedCompany>(KEYS.companies).filter(
    (c) => c.name.toLowerCase() !== clean.toLowerCase(),
  );
  const next: SavedCompany[] = [
    { id: newId(), name: clean, updatedAt: Date.now() },
    ...rest,
  ].slice(0, MAX_COMPANIES);
  write(KEYS.companies, next);
  return next;
}

export function removeCv(id: string): SavedCv[] {
  const next = read<SavedCv>(KEYS.cvs).filter((c) => c.id !== id);
  write(KEYS.cvs, next);
  return next;
}

export function removeJd(id: string): SavedJd[] {
  const next = read<SavedJd>(KEYS.jds).filter((j) => j.id !== id);
  write(KEYS.jds, next);
  return next;
}

export function removeCompany(id: string): SavedCompany[] {
  const next = read<SavedCompany>(KEYS.companies).filter((c) => c.id !== id);
  write(KEYS.companies, next);
  return next;
}
