export type Role = 'user' | 'assistant';

export interface BaseMessage {
  id: string;
  role: Role;
  timestamp: number;
}

export interface TextMessage extends BaseMessage {
  type: 'text';
  content: string;
}

export interface McqOption {
  id: string;
  text: string;
}

export interface McqMessage extends BaseMessage {
  type: 'mcq';
  question: string;
  options: McqOption[];
  selectedOptionId?: string;
}

export type Message = TextMessage | McqMessage;
