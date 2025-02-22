"use client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState, useEffect } from "react";
import io from "socket.io-client";
const socket = io("http://localhost:8000");
const Chat = () => {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  useEffect(() => {
    socket.on("message", (msg) => {
      setMessages((prev) => [...prev, { text: msg, type: "received" }]);
    });
    return () => {
      socket.off("message");
    };
  }, []);
  const sendMessage = () => {
    if (message.trim() === "") return;
    socket.emit("message", message);
    setMessages((prev) => [...prev, { text: message, type: "sent" }]);
    setMessage("");
  };
  return (
    <div className="w-full h-full flex flex-col gap-5">
      {messages.map((msg, index) =>
        msg.type === "sent" ? (
          <SentMessage key={index} message={msg.text} />
        ) : (
          <ReceivedMessage key={index} message={msg.text} />
        )
      )}

      <div className="absolute bottom-0 w-full text-white p-4 flex space-x-3 border-t border-input">
        <Input
          type="text"
          placeholder="How can Curiosity help you today ?"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <Button type="submit" onClick={sendMessage}>
          Send
        </Button>
      </div>
    </div>
  );
};
const SentMessage = ({ message }) => {
  return (
    <div className="self-end p-2 m-2 border border-input max-w-sm break-words">
      {message}
    </div>
  );
};
const ReceivedMessage = ({ message }) => {
  const [status, setStatus] = useState("Waiting");
  return (
    <div className="flex flex-col gap-2 self-start p-2 m-2 border-t border-input min-w-[90%] break-words">
      <div>
        <h1 className="m-1">{status}</h1>
        <h1 className="m-1 font-bold">Sources</h1>
        <div className="flex flex-wrap gap-2">
          <Sources url="https://www.google.com" />
          <Sources url="https://www.ndtv.com" />
          <Sources url="https://www.aneeshpatne.com" />
        </div>
        <div className="break-words p-2 font-medium">{message}</div>
      </div>
    </div>
  );
};
const Sources = ({ url }) => {
  const parsedUrl = url.replace(/^https?:\/\/www\./, "").split(".")[0];
  const parsedUrlforFavicon = url
    .replace(/^https?:\/\/(www\.)?/, "")
    .split("/")[0];
  return (
    <div className="flex items-center gap-2 border border-input p-1">
      <img
        src={`https://www.google.com/s2/favicons?domain=${parsedUrlforFavicon}`}
        alt="favicon"
        className="w-[20px] h-[20px] object-cover"
      />
      {parsedUrl}
    </div>
  );
};
export default Chat;
