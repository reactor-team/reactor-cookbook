import { HeliosApp } from "../HeliosApp";
import { SetupRequired } from "../SetupRequired";

// The original create-reactor-app Helios demo, preserved unchanged at
// /demo. The Song World prototype lives at the root route.
export const dynamic = "force-dynamic";

export default function DemoPage() {
  const hasKey = !!process.env.REACTOR_API_KEY;
  return hasKey ? <HeliosApp /> : <SetupRequired />;
}
