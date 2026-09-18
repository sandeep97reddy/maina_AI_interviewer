import { redirect } from "next/navigation";

/**
 * The setup form IS the homepage. Every `href="/"` in the app (headers,
 * report page, avatars page) lands here and continues to /setup, so no
 * caller needed changes.
 */
export default function Home() {
  redirect("/setup");
}
