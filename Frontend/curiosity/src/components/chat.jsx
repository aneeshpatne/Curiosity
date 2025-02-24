"use client";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import io from "socket.io-client";
import { v4 as uuidv4 } from "uuid";
import { Bot } from "lucide-react";
import parse from "html-react-parser";
import DOMPurify from "dompurify";
import { marked } from "marked";
const socket = io("http://localhost:4000");

const Chat = () => {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [searchType, setSearchType] = useState("normal");

  // Handle receiving sources
  useEffect(() => {
    socket.on("sources", (data) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === data.id && msg.type === "received"
            ? { ...msg, sources: data.sources }
            : msg
        )
      );
    });

    // Handle receiving messages
    socket.on("message", (data) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === data.id && msg.type === "received"
            ? { ...msg, text: data.text, status: data.status }
            : msg
        )
      );
    });

    // Handle status updates
    socket.on("status", (data) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === data.id && msg.type === "received"
            ? { ...msg, status: data.status }
            : msg
        )
      );
    });

    return () => {
      socket.off("sources");
      socket.off("message");
      socket.off("status");
    };
  }, []);

  // Function to send messages
  const sendMessage = () => {
    if (!message.trim()) return;

    const id = uuidv4();
    setMessages((prev) => [
      ...prev,
      { id, text: message, type: "sent" },
      { id, text: "", type: "received", sources: [], status: "pending" },
    ]);

    socket.emit("message", { id, text: message });
    setMessage(""); // Clear input after sending
  };

  // Handle send event (enter key or button click)
  const handleSend = () => {
    sendMessage();
  };

  return (
    <>
      {messages.length === 0 ? (
        <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
          <Bot className="w-12 h-12 text-primary animate-pulse" />
          <h1 className="text-4xl font-semibold text-center">
            Hi, how can Curiosity help you?
          </h1>
        </div>
      ) : (
        <div className="w-full flex flex-col gap-5 mb-24">
          {messages.map((msg) =>
            msg.type === "sent" ? (
              <SentMessage key={`${msg.id}-sent`} message={msg.text} />
            ) : (
              <ReceivedMessage
                key={`${msg.id}-received`}
                message={msg.text}
                sources={msg.sources}
                status={msg.status}
              />
            )
          )}
        </div>
      )}

      {/* Chat Input */}
      <div className="fixed bottom-0 w-full text-white p-4 flex space-x-3 border-t border-input bg-background">
        <Select value={searchType} onValueChange={setSearchType}>
          <SelectTrigger className="w-[180px]">
            {" "}
            {/* Increased width */}
            <SelectValue placeholder="Search Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="normal">Normal Search</SelectItem>
            <SelectItem value="pro">Pro Search</SelectItem>
          </SelectContent>
        </Select>

        <Input
          type="text"
          placeholder="Ask Curiosity Anything..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          className="flex-grow" // Make input take remaining space
        />
        <Button type="submit" onClick={handleSend}>
          Send
        </Button>
      </div>
    </>
  );
};
const SentMessage = ({ message }) => {
  return (
    <div className="self-end p-2 m-2 border border-input max-w-sm break-words">
      {message}
    </div>
  );
};

const Citation = ({ number }) => {
  return (
    <a
      href={`#source-${number}`} // Link to source section
      className="relative group text-gray-400 hover:text-gray-300 transition-all"
    >
      <sup className="px-0.5 py-0.10 bg-gray-700 text-gray-300 rounded-sm text-xs group-hover:bg-gray-600 group-hover:text-white shadow-sm">
        {number}
      </sup>
    </a>
  );
};

const MarkdownRenderer = ({ content }) => {
  // Convert markdown to sanitized HTML first
  let rawHtml = DOMPurify.sanitize(marked(content));

  // Replace citations ([1], [2]) with a unique marker
  rawHtml = rawHtml.replace(
    /\[(\d+)\]/g,
    `<span class="citation" data-cite="$1">[$1]</span>`
  );

  // Function to transform <span class="citation"> into <Citation /> component
  const transform = (node) => {
    if (
      node.type === "tag" &&
      node.name === "span" &&
      node.attribs?.class === "citation"
    ) {
      return <Citation number={node.attribs["data-cite"]} />;
    }
    return undefined; // Default processing
  };

  return <div>{parse(rawHtml, { replace: transform })}</div>;
};

const ReceivedMessage = ({ message, sources, status }) => {
  const [statusText, setStatusText] = useState(status);

  useEffect(() => {
    setStatusText(status);
  }, [status]);

  return (
    <div className="flex flex-col gap-2 self-start p-2 m-2 border-t border-input w-[90%] break-words">
      {statusText !== "finished" && (
        <div className="font-light italic">{statusText}...</div>
      )}

      <div>
        {sources && sources.length > 0 && (
          <p className="m-1 text-lg font-light tracking-wide text-primary">
            Sources:
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          {sources?.map((url, index) => (
            <Sources key={index} url={url} />
          ))}
        </div>

        <div className="break-words p-2 font-medium markdown-content">
          <MarkdownRenderer content={message} />
        </div>
      </div>
    </div>
  );
};

const getDomainName = (url) => {
  try {
    const parsedUrl = new URL(url);
    const hostnameParts = parsedUrl.hostname.split(".");

    // Handle cases like "www.example.com" and "example.com"
    if (hostnameParts.length > 2) {
      return hostnameParts[hostnameParts.length - 2]; // Extract second-last part (e.g., "example")
    }
    return hostnameParts[0]; // Fallback for cases like "example.com"
  } catch (error) {
    console.error("Invalid URL:", url);
    return url; // Fallback: return the full URL if parsing fails
  }
};

const Sources = ({ url }) => {
  const parsedUrl = getDomainName(url);
  const parsedUrlForFavicon = url
    .replace(/^https?:\/\/(www\.)?/, "")
    .split("/")[0];
  return (
    <div className="flex items-center gap-2 border border-input p-1">
      <img
        src={`https://www.google.com/s2/favicons?domain=${parsedUrlForFavicon}`}
        alt="favicon"
        className="w-[20px] h-[20px] object-cover"
      />
      {parsedUrl}
    </div>
  );
};

export default Chat;
