import gradio as gr
from pipeline.inference_pipeline import prediction


with gr.Blocks(title="Spam Classifier Showdown", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚔️ AI Model Showdown: Spam Detection")
    gr.Markdown("Compare the performance of a **Custom Bi-LSTM** vs. **DistilBERT** in real-time.")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter SMS or Email Content", 
            placeholder="Type your message here (e.g., 'Free money now!')...",
            lines=2
        )
        
    run_btn = gr.Button("🚀 Analyze Message", variant="primary")
    
    with gr.Row():
        # Column 1: LSTM
        with gr.Column():
            gr.Markdown("### 🧠 Custom Bi-LSTM (Tiny & Fast)")
            lstm_label = gr.Label(label="Prediction")
            lstm_conf = gr.Textbox(label="Confidence")
            lstm_time = gr.Textbox(label="Inference Time")
            
        # Column 2: BERT
        with gr.Column():
            gr.Markdown("### 🤖 DistilBERT (Large & Smart)")
            bert_label = gr.Label(label="Prediction")
            bert_conf = gr.Textbox(label="Confidence")
            bert_time = gr.Textbox(label="Inference Time")

    # Click Event
    run_btn.click(
        fn=prediction,
        inputs=input_text,
        outputs=[lstm_label, lstm_conf, lstm_time, bert_label, bert_conf, bert_time]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"],
            ["Hey, are we still meeting for lunch tomorrow?"],
            ["Your account has been compromised. Click here to reset your password."],
            ["I'm forcing myself to eat a slice. I'm really not hungry tho."]
        ],
        inputs=input_text
    )

# Launch
demo.launch()