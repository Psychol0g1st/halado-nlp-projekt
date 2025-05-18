import React, { useRef, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getImagesFromMessage, getTextContentFromMessage } from "@/lib/utils";
import Image from "next/image";
import CodeDisplayBlock from "../code-display-block";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "../ui/button";
import { ChatProps } from "@/lib/types";
import {
  CheckIcon,
  CopyIcon,
  FileTextIcon,
  RefreshCcw,
  Volume2,
  VolumeX,
} from "lucide-react";
import useChatStore from "@/hooks/useChatStore";
import ButtonWithTooltip from "../button-with-tooltip";
import { ChatMessageList } from "../ui/chat/chat-message-list";
import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "../ui/chat/chat-bubble";

export default function ChatList({
  messages,
  loadingSubmit,
  onRegenerate,
}: ChatProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [name, setName] = React.useState<string>("");
  const [localStorageIsLoading, setLocalStorageIsLoading] =
    React.useState(true);
  const [isCopied, setisCopied] = React.useState<Record<number, boolean>>({});
  const [textToSpeech, setTextToSpeech] =
    useState<SpeechSynthesisUtterance | null>(null);
  const [isSpeaking, setIsSpeaking] = React.useState<Record<number, boolean>>(
    {}
  );
  const [currentSpeakingIndex, setCurrentSpeakingIndex] = useState<
    number | null
  >(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const utterance = new SpeechSynthesisUtterance();
      utterance.volume = 0.2;

      const setVoice = () => {
        const voices = speechSynthesis.getVoices();
        utterance.voice = voices[6];
        setTextToSpeech(utterance);
      };

      setVoice();

      speechSynthesis.addEventListener("voiceschanged", setVoice);

      return () => {
        speechSynthesis.cancel();
        speechSynthesis.removeEventListener("voiceschanged", setVoice);
      };
    }
  }, []);

  // Zustand
  const isLoading = useChatStore((state) => state.isLoading);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const username = localStorage.getItem("chatty_user");
      if (username) {
        setName(username);
        setLocalStorageIsLoading(false);
      }
    }
  }, []);

  const copyToClipboard = (response: string, index: number) => () => {
    navigator.clipboard.writeText(response);
    setisCopied((prevState) => ({ ...prevState, [index]: true }));
    setTimeout(() => {
      setisCopied((prevState) => ({ ...prevState, [index]: false }));
    }, 1500);
  };

  const handleTextToSpeech = (text: string, index: number) => {
    if (!textToSpeech) return;
    // Stop the currently speaking text if any
    if (currentSpeakingIndex !== null) {
      speechSynthesis.cancel();
      setIsSpeaking((prevState) => ({
        ...prevState,
        [currentSpeakingIndex]: false,
      }));
    }
    // Start the new text-to-speech
    if (isSpeaking[index]) {
      speechSynthesis.cancel();
      setIsSpeaking((prevState) => ({ ...prevState, [index]: false }));
      setCurrentSpeakingIndex(null);
    } else {
      textToSpeech.text = text;
      speechSynthesis.speak(textToSpeech);
      setIsSpeaking((prevState) => ({ ...prevState, [index]: true }));
      setCurrentSpeakingIndex(index);

      textToSpeech.onend = () => {
        setIsSpeaking((prevState) => ({ ...prevState, [index]: false }));
        setCurrentSpeakingIndex(null);
      };
    }
  };

  const getThinkContent = (content: string) => {
    const match = content.match(/<think>([\s\S]*?)(?:<\/think>|$)/);
    return match ? match[1].trim() : null;
  };

  return (
    <div className="flex-1 w-full overflow-y-auto">
      <ChatMessageList>
      {messages.map((msg, idx) => {
        const isUser = idx % 2 === 0; // user sends every other message

        return (
          <div
          key={idx}
          className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
          >
            <div
            className={`${
              isUser
              ? 'bg-blue-500 text-white rounded-2xl rounded-br-none max-w-sm'
              : 'bg-gray-200 text-gray-900 rounded-2xl rounded-bl-none w-full'
              } px-4 py-2 text-sm`}
            >
              {msg}
            </div>
          </div>
        );
      })}
      {loadingSubmit && (
        <div className="flex justify-start">
          <div className="w-full bg-gray-200 text-gray-600 px-4 py-2 rounded-2xl rounded-bl-none text-sm max-w-sm">
            Loading...
          </div>
        </div>
      )}
      </ChatMessageList>
    </div>
  );
}
