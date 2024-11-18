import React, { useState } from 'react';
import '../../css/Chatbot.css'; 

const Chatbot = () => {
  // console.log("entered Chatbot");
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]); // History of chat sessions

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    // Add user message to the chat
    const userMessage = { text: inputText, sender: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    setLoading(true);
    setInputText(''); // Clear the input field

    try {
      const res = await fetch('http://localhost:8000/legal-advice', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputText }), // Send the query from inputText
      });

      const data = await res.json();
      // console.log(data);
      const botMessage = { text: data.result, sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);

      // Add chat session to history once it's complete
      setHistory((prevHistory) => [...prevHistory, { user: inputText, bot: data.result.result }]);
    } catch (error) {
      console.error('Error fetching legal advice:', error);
      const errorMessage = { text: 'An error occurred while fetching the legal advice.', sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chatbot-container">
      {/* Left side - Chat */}
      <div className="chatbot-content">
        <div className="chatbot-messages">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.sender === 'bot' ? 'bot-message' : 'user-message'}`}
            >
              {message.text}
            </div>
          ))}
        </div>
        <div className="chatbot-input">
          <input
            type="text"
            placeholder="Type your message..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleSubmit(e); // Pass the event to handleSubmit
              }
            }}
          />
          <button onClick={handleSubmit}>Send</button>
        </div>
      </div>

      {/* Right side - Chat history */}
      <div className="chatbot-history">
        <h2>Chat History</h2>
        {history.map((session, index) => (
          <div key={index} className="history-item">
            <p> {session.user}</p>
            {/* <p><strong>Bot:</strong> {session.bot}</p> */}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Chatbot;
