import os

from collective_agent import process_collective_jobs
from dotenv import load_dotenv
from mail_agent import process_emails
from social_agent import process_social_jobs
from video_agent import process_video_jobs

load_dotenv()


def run_agent():
    agent_type = os.getenv("URKS_AGENT_TYPE", "mail").strip().lower()
    if agent_type == "video":
        process_video_jobs()
        return
    if agent_type == "collective":
        process_collective_jobs()
        return
    if agent_type == "social":
        process_social_jobs()
        return
    process_emails()


if __name__ == "__main__":
    run_agent()
