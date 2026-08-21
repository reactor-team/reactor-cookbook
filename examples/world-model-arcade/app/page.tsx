import { ArcadeExperience } from "@/components/ArcadeExperience";

export const dynamic = "force-dynamic";

export default function Page() {
  return <ArcadeExperience configured={Boolean(process.env.REACTOR_API_KEY)} />;
}
