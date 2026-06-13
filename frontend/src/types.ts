export type Role = 'user' | 'assistant';

export interface BaseMessage {
  id: string;
  role: Role;
  timestamp: number;
}

export interface TextMessage extends BaseMessage {
  type: 'text';
  content: string;
  /** SQL query used to generate this response (assistant only) */
  sql?: string | null;
  /** Intent classification: READ | GENERAL | WRITE_BLOCKED | AMBIGUOUS */
  intent?: string;
  /** Number of DB rows returned */
  row_count?: number | null;
  /** CSV filename served at /download/{csv_path} */
  csv_path?: string | null;
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
