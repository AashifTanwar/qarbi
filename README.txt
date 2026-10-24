QARBI AI — UNIVERSAL RESEARCH KNOWLEDGE GRAPH

Bundle files:
- index.html: Qarbi AI interface with universal topic research, interactive knowledge graph, source links, uploads, and local research history.
- main.py: FastAPI backend with existing upload/chat endpoints plus /api/research and /api/expand.

INSTALL
1. Back up your current index.html and backend main.py.
2. Copy qarbi_ai_universal_graph.html to your website folder and rename it index.html.
3. Copy qarbi_ai_main_universal.py to your backend folder and rename it main.py.
4. Keep your private API key in backend/.env. Configure GEMINI_API_KEY and GEMINI_MODEL to a model actually available to your Google AI API project that supports Google Search grounding. Do not paste the key into chats or source files.
5. Ensure the backend environment has the dependencies already used by your current project, including fastapi, uvicorn, python-multipart, google-genai, python-dotenv, and document/OCR libraries used by your existing upload setup.
6. Restart the backend using your existing launch command, then open http://localhost:5173.

HOW IT WORKS
- Enter any topic or question to create an initial graph.
- Select a node and click “Explore further”. First click reveals existing children; when already expanded, the button researches more detail and appends new nodes. Double-clicking a node also invokes expansion.
- The graph structure adapts to the subject; it is not restricted to colleges.
- Research requires internet access, an API key, and a Gemini model with Google Search grounding. Model availability depends on your API project.
- Source links are shown when the provider returns them. AI research is not guaranteed exhaustive; verify important facts.
- Browser history is stored locally and is not synced between devices.

TEST
- Python syntax check: run `python -m py_compile main.py`.
- Check `http://127.0.0.1:8000/api/health` and `http://127.0.0.1:8000/docs`.
- Try a universal query such as “How does a lithium-ion battery work?” then select a node and choose Explore further.
