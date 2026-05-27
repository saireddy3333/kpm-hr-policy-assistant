"""Browser UI and smoke test entry point for the KPM HR Policy Assistant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import config
from rag.chunker import save_chunks
from rag.document_loader import save_clean_text
from rag.embeddings import build_index, ensure_index
from rag.generator import HRPolicyAssistant
from rag.memory import ConversationMemory


SMOKE_QUESTIONS = [
    ("How does FMLA work?", {"5.6"}),
    ("What happens if I'm late?", {"3.2", "10.6"}),
    ("Can I work from home?", {"3.4"}),
    ("What PPE do I need?", {"6.5", "7.7"}),
    ("What is the drug policy?", {"6.3"}),
]


def rebuild_pipeline() -> str:
    save_clean_text()
    save_chunks()
    meta = build_index(rebuild=True)
    return (
        f"Rebuilt {meta['chunk_count']} chunks using {meta['backend']} "
        f"embeddings and {meta['index_backend']} index."
    )


def ensure_pipeline() -> None:
    ensure_index()


def sources_markdown(sources: list[dict]) -> str:
    if not sources:
        return "No source sections retrieved."
    lines = ["### Retrieved Sources"]
    for source in sources:
        lines.append(
            f"- **Section {source['subsection']} {source['title']}**  \n"
            f"  `{source['chunk_id']}` | confidence {source['confidence']}  \n"
            f"  {source['snippet']}"
        )
    return "\n".join(lines)


def run_smoke_test(write_report: bool = True) -> list[dict]:
    ensure_pipeline()
    assistant = HRPolicyAssistant()
    memory = ConversationMemory()
    outputs: list[dict] = []
    for question, expected_sections in SMOKE_QUESTIONS:
        result = assistant.answer(question, name="Alex", memory=memory)
        top3 = result.retrieved_sections[:3]
        hit = bool(expected_sections & set(top3))
        item = {
            "question": question,
            "expected_sections": sorted(expected_sections),
            "retrieved_sections": result.retrieved_sections,
            "top3_hit": hit,
            "answer": result.answer,
            "sources": result.sources,
        }
        outputs.append(item)
        print("=" * 80)
        print(f"User question: {question}")
        print(f"Retrieved sections: {', '.join(result.retrieved_sections) or 'None'}")
        print(f"Expected section appeared in top 3: {hit}")
        print("Final chatbot answer:")
        print(result.answer)

    if write_report:
        write_test_report(outputs)
    return outputs


def write_test_report(smoke_outputs: list[dict]) -> Path:
    metrics = None
    results_path = config.EVALUATION_DIR / "results.json"
    if results_path.exists():
        metrics = json.loads(results_path.read_text(encoding="utf-8"))

    lines = ["# TEST REPORT", ""]
    if metrics:
        lines.extend(
            [
                "## Evaluation Results",
                f"- HITS@1: {metrics.get('hits_at_1', 0):.3f}",
                f"- HITS@3: {metrics.get('hits_at_3', 0):.3f}",
                f"- Intent accuracy: {metrics.get('intent_accuracy', 0):.3f}",
                f"- Intent macro F1: {metrics.get('intent_macro_f1', 0):.3f}",
                "",
            ]
        )

    lines.extend(["## Smoke Test Outputs", ""])
    for item in smoke_outputs:
        source_citations = [
            f"Section {source['subsection']} {source['title']}" for source in item["sources"]
        ]
        lines.extend(
            [
                f"### {item['question']}",
                f"- Retrieved sections: {', '.join(item['retrieved_sections']) or 'None'}",
                f"- Source citations: {'; '.join(source_citations) or 'None'}",
                f"- Expected section appeared in top 3: {item['top3_hit']}",
                "",
                "Final chatbot answer:",
                "",
                item["answer"],
                "",
            ]
        )

    report_path = config.PROJECT_ROOT / "TEST_REPORT.md"
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


def build_interface():
    import gradio as gr

    ensure_pipeline()
    assistant = HRPolicyAssistant()

    def submit(user_message, chat_history, memory_state, name):
        if not user_message or not user_message.strip():
            return chat_history, memory_state, "", "Enter a question to search KPM policies.", ""
        memory = ConversationMemory.from_state(memory_state)
        result = assistant.answer(user_message, name=name, memory=memory)
        chat_history = (chat_history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": result.answer},
        ]
        return (
            chat_history,
            memory.to_state(),
            "",
            "Searched KPM policies.",
            sources_markdown(result.sources),
        )

    def quick_submit(question, chat_history, memory_state, name):
        return submit(question, chat_history, memory_state, name)

    def clear_chat():
        return [], [], "Conversation cleared.", "No source sections retrieved."

    def rebuild():
        status = rebuild_pipeline()
        return status

    with gr.Blocks(title="KPM HR Policy Assistant") as demo:
        gr.Markdown(
            "# KPM HR Policy Assistant\n"
            "Ask me questions about KPM HR policies. I answer using the official policy manual and show sources."
        )
        memory_state = gr.State([])
        with gr.Row():
            name = gr.Textbox(label="Name", value="Alex", scale=1)
            status = gr.Textbox(label="Status", value="Ready to search KPM policies.", interactive=False, scale=2)
        chatbot = gr.Chatbot(label="Chat", type="messages", height=430, allow_tags=False)
        with gr.Row():
            message = gr.Textbox(
                label="Ask a question",
                placeholder="Example: How does FMLA work?",
                scale=5,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        gr.Markdown("### Quick Questions")
        quick_questions = [
            "How does FMLA work?",
            "What happens if I'm late?",
            "Can I work from home?",
            "What PPE do I need?",
            "What is the dress code?",
            "What is the drug policy?",
            "How does overtime work?",
            "What if my paycheck is wrong?",
            "How do performance reviews work?",
        ]
        sources = gr.Markdown("No source sections retrieved.", label="Source Citations")
        with gr.Row():
            for question in quick_questions[:3]:
                gr.Button(question).click(
                    quick_submit,
                    inputs=[gr.State(question), chatbot, memory_state, name],
                    outputs=[chatbot, memory_state, message, status, sources],
                )
        with gr.Row():
            for question in quick_questions[3:6]:
                gr.Button(question).click(
                    quick_submit,
                    inputs=[gr.State(question), chatbot, memory_state, name],
                    outputs=[chatbot, memory_state, message, status, sources],
                )
        with gr.Row():
            for question in quick_questions[6:]:
                gr.Button(question).click(
                    quick_submit,
                    inputs=[gr.State(question), chatbot, memory_state, name],
                    outputs=[chatbot, memory_state, message, status, sources],
                )
        with gr.Row():
            clear = gr.Button("Clear conversation")
            rebuild_button = gr.Button("Rebuild index")

        send.click(
            submit,
            inputs=[message, chatbot, memory_state, name],
            outputs=[chatbot, memory_state, message, status, sources],
        )
        message.submit(
            submit,
            inputs=[message, chatbot, memory_state, name],
            outputs=[chatbot, memory_state, message, status, sources],
        )
        clear.click(clear_chat, outputs=[chatbot, memory_state, status, sources])
        rebuild_button.click(rebuild, outputs=[status])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KPM HR Policy Assistant.")
    parser.add_argument("--smoke-test", action="store_true", help="Run chatbot pipeline without launching Gradio.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the policy index before launching.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.rebuild:
        print(rebuild_pipeline())

    if args.smoke_test:
        run_smoke_test(write_report=True)
        return

    demo = build_interface()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
