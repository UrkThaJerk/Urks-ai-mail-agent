import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI


COLLECTIVE_MODEL = os.getenv("COLLECTIVE_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class SpecialistAgent:
    name: str
    specialty: str
    system_prompt: str


@dataclass
class SpecialistReport:
    name: str
    specialty: str
    independent_work: str
    peer_learning: str


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CollectiveLearningAgent:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or get_openai_client()
        self.specialists = [
            SpecialistAgent(
                name="mail_specialist",
                specialty="email triage, drafting, and inbox automation",
                system_prompt=(
                    "You are a mail specialist in a 3-agent collective. "
                    "Work independently first, then help the other agents improve. "
                    "Be concrete, concise, and operational."
                ),
            ),
            SpecialistAgent(
                name="video_specialist",
                specialty="video editing, scene analysis, and media export workflows",
                system_prompt=(
                    "You are a video specialist in a 3-agent collective. "
                    "Work independently first, then help the other agents improve. "
                    "Be concrete, concise, and operational."
                ),
            ),
            SpecialistAgent(
                name="strategy_specialist",
                specialty="cross-agent orchestration, feedback loops, and continuous improvement",
                system_prompt=(
                    "You are a strategy specialist in a 3-agent collective. "
                    "Work independently first, then help the other agents improve. "
                    "Be concrete, concise, and operational."
                ),
            ),
        ]

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=COLLECTIVE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = response.choices[0].message
        return (message["content"] if isinstance(message, dict) else message.content or "").strip()

    def _build_independent_prompt(self, specialist: SpecialistAgent, objective: str, context: str) -> str:
        return (
            f"Objective: {objective}\n"
            f"Shared context: {context or 'None provided.'}\n"
            f"Your specialty: {specialist.specialty}\n\n"
            "Work separately from the other two agents. Provide:\n"
            "1. Your main contribution\n"
            "2. The biggest risk you see\n"
            "3. What the other agents should know from your perspective"
        )

    def _build_peer_learning_prompt(
        self,
        specialist: SpecialistAgent,
        objective: str,
        context: str,
        independent_work: str,
        peer_reports: list[SpecialistReport],
    ) -> str:
        peer_summary = "\n\n".join(
            f"{peer.name} ({peer.specialty}):\n{peer.independent_work}" for peer in peer_reports
        )
        return (
            f"Objective: {objective}\n"
            f"Shared context: {context or 'None provided.'}\n"
            f"Your specialty: {specialist.specialty}\n"
            f"Your independent work:\n{independent_work}\n\n"
            f"Peer reports:\n{peer_summary}\n\n"
            "Now come together collectively. Explain what you learned from the other agents and "
            "how their ideas make your own work better."
        )

    def _build_collective_prompt(self, objective: str, context: str, reports: list[SpecialistReport]) -> str:
        combined_reports = "\n\n".join(
            (
                f"{report.name} ({report.specialty})\n"
                f"Independent work:\n{report.independent_work}\n"
                f"Peer learning:\n{report.peer_learning}"
            )
            for report in reports
        )
        return (
            f"Objective: {objective}\n"
            f"Shared context: {context or 'None provided.'}\n\n"
            f"Agent reports:\n{combined_reports}\n\n"
            "Create a concise collective summary with:\n"
            "1. Shared plan\n"
            "2. What each agent learned from the others\n"
            "3. The strongest feedback loop to keep improving future work"
        )

    def run_collective(self, objective: str, context: str = "") -> dict[str, object]:
        reports = [
            SpecialistReport(
                name=specialist.name,
                specialty=specialist.specialty,
                independent_work=self._complete(
                    specialist.system_prompt,
                    self._build_independent_prompt(specialist, objective, context),
                ),
                peer_learning="",
            )
            for specialist in self.specialists
        ]

        completed_reports: list[SpecialistReport] = []
        for specialist, report in zip(self.specialists, reports):
            peer_reports = [peer for peer in reports if peer.name != specialist.name]
            report.peer_learning = self._complete(
                specialist.system_prompt,
                self._build_peer_learning_prompt(
                    specialist,
                    objective,
                    context,
                    report.independent_work,
                    peer_reports,
                ),
            )
            completed_reports.append(report)

        collective_summary = self._complete(
            (
                "You are the synthesis layer for a 3-agent collective. "
                "Combine their work into a shared outcome and emphasize continuous learning."
            ),
            self._build_collective_prompt(objective, context, completed_reports),
        )

        return {
            "objective": objective,
            "context": context,
            "agents": [asdict(report) for report in completed_reports],
            "collective_summary": collective_summary,
        }


def process_collective_jobs() -> None:
    objective = os.getenv("COLLECTIVE_OBJECTIVE", "").strip()
    if not objective:
        raise ValueError("Set COLLECTIVE_OBJECTIVE to run the three-agent collaborative workflow.")

    context = os.getenv("COLLECTIVE_CONTEXT", "").strip()
    output_path = os.getenv("COLLECTIVE_OUTPUT_PATH", "").strip()
    result = CollectiveLearningAgent().run_collective(objective, context)

    if output_path:
        resolved_path = Path(output_path).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    process_collective_jobs()
