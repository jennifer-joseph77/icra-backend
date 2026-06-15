const form = document.getElementById("chatForm");
const input = document.getElementById("questionInput");
const chatContainer = document.getElementById("chatContainer");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";

  const loading = addMessage("ICRA is typing...", "bot");

  try {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question })
    });

    const data = await response.json();
    loading.remove();

    const botMessage = addMessage(data.answer, "bot");

    if (data.sources && data.sources.length > 0) {
      const sources = document.createElement("div");
      sources.className = "sources";
      sources.textContent =
        "Sources: " + data.sources.map(s => s.name).join(", ");
      botMessage.appendChild(sources);
    }

  } catch (error) {
    loading.remove();
    addMessage("Error connecting to server.", "bot");
  }
});

function addMessage(text, type) {
  const msg = document.createElement("div");
  msg.classList.add("message", type);
  msg.textContent = text;
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return msg;
}

function startNewChat() {
  chatContainer.innerHTML = `
    <div class="welcome">
      <h1>ICRA</h1>
      <p>How can I help you today?</p>
    </div>
  `;
}
