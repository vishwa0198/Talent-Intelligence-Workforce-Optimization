# Talent Intelligence & Workforce Optimization Platform

An AI-powered HR assistant that leverages OpenAI's advanced language models to streamline critical human resources workflows, from candidate evaluation to employee feedback analysis.

## 🚀 Features

### 1. **Intelligent HR Chatbot**
- Conversational AI assistant for HR policy questions
- Context-aware multi-turn dialogues
- Professional, concise responses with step-by-step guidance

### 2. **Interview Analysis**
- Automatic transcript summarization (PDF/TXT support)
- Top 5 skills extraction
- Strengths & weaknesses identification
- Data-driven hiring recommendations

### 3. **Resume Optimization**
- Comprehensive resume feedback
- ATS-friendliness scoring (0-100)
- Missing keywords detection for target roles
- Bullet point rewrites (before/after format)
- Gap analysis and improvement suggestions

### 4. **Job Description Generator**
- ATS-optimized job postings
- Structured format: title, summary, responsibilities, qualifications
- Customizable by seniority level (Junior/Mid/Senior)
- Consistent quality across all recruitment materials

### 5. **Employee Review Analytics**
- Sentiment analysis (Positive/Neutral/Negative)
- Actionable HR insights extraction
- Professional response drafting
- Proactive workplace culture improvement

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see Installation)

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install openai pymupdf python-dotenv sentence-transformers
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

Or set the environment variable directly:

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

**Windows:**
```bash
set OPENAI_API_KEY=your_openai_api_key_here
```

## 🎯 Usage

### Interactive CLI Mode

Run the chatbot in interactive mode:

```bash
python chatbot.py
```

You'll see a menu with 6 options:

1. **Chat** - Ask HR questions interactively
2. **Summarize transcript** - Analyze interview transcripts
3. **Resume feedback** - Get resume improvement suggestions
4. **Generate JD** - Create job descriptions
5. **Analyze review** - Process employee reviews
6. **Exit** - Close the application

### Python API Usage

```python
from chatbot import (
    chat_with_hr_bot,
    summarize_interview_transcript,
    resume_improvement_feedback,
    generate_job_description,
    analyze_employee_review,
    safe_read_pdf_bytes,
    safe_read_txt
)

# Example 1: Chat with HR bot
response = chat_with_hr_bot("What is the company's PTO policy?")
print(response["response"])

# Example 2: Summarize interview
transcript = safe_read_txt("interview_transcript.txt")
summary = summarize_interview_transcript(transcript, summary_length="detailed")
print(summary)

# Example 3: Resume feedback
resume = safe_read_pdf_bytes("candidate_resume.pdf")
feedback = resume_improvement_feedback(resume, role_target="Senior Software Engineer")
print(feedback)

# Example 4: Generate job description
jd = generate_job_description(
    role_name="Data Scientist",
    skills_bullets="Python, Machine Learning, SQL, Statistics",
    seniority="Senior"
)
print(jd)

# Example 5: Analyze employee review
analysis = analyze_employee_review("Great company culture but work-life balance needs improvement...")
print(analysis)
```

## 📁 Project Structure

```
.
├── chatbot.py           # Main application with all HR AI functions
├── .env                 # Environment variables (API keys)
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── data/               # Data folder (ignored by git)
│   ├── resumes/
│   ├── transcripts/
│   └── reviews/
└── requirements.txt    # Python dependencies (optional)
```

## 🔧 Configuration

### Model Selection

Change the default model in `chatbot.py`:

```python
DEFAULT_MODEL = "gpt-4"  # Options: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"
DEFAULT_TEMPERATURE = 0.3  # 0.0 = deterministic, 1.0 = creative
```

### Custom System Prompts

Customize the HR assistant behavior:

```python
custom_prompt = "You are an HR specialist for a tech startup..."
response = chat_with_hr_bot(
    "How do I request parental leave?",
    system_prompt=custom_prompt
)
```

## 📊 Supported File Formats

- **PDF** - Resumes, transcripts, documents (via PyMuPDF)
- **TXT** - Plain text files
- **Future**: DOCX, CSV support planned

## ⚠️ Error Handling

The application includes:
- Automatic retry logic (2 retries with exponential backoff)
- Graceful PDF extraction failures
- API key validation warnings
- UTF-8 encoding error handling

## 🔒 Security Best Practices

1. **Never commit `.env` files** - Already in `.gitignore`
2. **Rotate API keys regularly**
3. **Use environment variables** for sensitive data
4. **Review data privacy** before processing employee information
5. **Comply with GDPR/local regulations** when handling personal data

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### "OPENAI_API_KEY not set" warning
- Ensure `.env` file exists with valid API key
- Or set environment variable before running

### PDF extraction fails
- Install PyMuPDF: `pip install pymupdf`
- Verify PDF is not password-protected

### Import errors
- Activate virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

## 📧 Support

For issues, questions, or feature requests, please open an issue on GitHub.

## 🙏 Acknowledgments

- OpenAI for GPT models
- PyMuPDF for PDF processing
- Sentence Transformers for embeddings (optional)

---

**Built with ❤️ for HR professionals who want to work smarter, not harder.**