from enum import IntEnum
from datetime import datetime
from typing import Dict, List
import os

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Table,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    scoped_session,
)

from dotenv import load_dotenv
import telebot
from telebot import types
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import TextSearchIndexType

Base = declarative_base()
engine = create_engine("sqlite:///db.sqlite3", echo=False, future=True)
SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False)
)

class Role(IntEnum):
    INTERN = 0
    JUNIOR = 1
    MIDDLE = 2
    SENIOR = 3
    LEAD = 4

ticket_tag = Table(
    "ticket_tag",
    Base.metadata,
    Column("ticket_id", Integer, ForeignKey("tickets.id")),
    Column("tag_id", Integer, ForeignKey("tags.id")),
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    tg_id = Column(Integer, unique=True, index=True)
    role = Column(Integer, default=int(Role.INTERN))

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True)
    description = Column(Text)
    solution = Column(Text, nullable=True)
    min_role = Column(Integer, default=int(Role.INTERN))
    author_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.utcnow)
    tags = relationship("Tag", secondary=ticket_tag)

def init_db():
    Base.metadata.create_all(bind=engine)

class UserService:
    @staticmethod
    def get(tg_id: int) -> User:
        with SessionLocal() as s:
            user = s.query(User).filter_by(tg_id=tg_id).first()
            if not user:
                user = User(tg_id=tg_id)
                s.add(user)
                s.commit()
                s.refresh(user)
            return user

    @staticmethod
    def set_role(user: User, role: Role):
        with SessionLocal() as s:
            db_user = s.query(User).get(user.id)
            db_user.role = int(role)
            s.commit()

class TagService:
    @staticmethod
    def create(name: str) -> Tag:
        with SessionLocal() as s:
            tag = Tag(name=name)
            s.add(tag)
            s.commit()
            s.refresh(tag)
            return tag

    @staticmethod
    def all() -> List[Tag]:
        with SessionLocal() as s:
            return s.query(Tag).all()

    @staticmethod
    def get(name: str) -> Tag | None:
        with SessionLocal() as s:
            return s.query(Tag).filter_by(name=name).first()

class TicketService:
    @staticmethod
    def create(author: User, description: str, min_role: Role) -> Ticket:
        with SessionLocal() as s:
            ticket = Ticket(
                description=description,
                author_id=author.id,
                min_role=int(min_role),
            )
            s.add(ticket)
            s.commit()
            s.refresh(ticket)
            return ticket

    @staticmethod
    def create_with_solution(
        author: User, description: str, solution: str, min_role: Role
    ) -> Ticket:
        with SessionLocal() as s:
            ticket = Ticket(
                description=description,
                author_id=author.id,
                solution=solution,
                status="closed",
                min_role=int(min_role),
            )
            s.add(ticket)
            s.commit()
            s.refresh(ticket)
            return ticket

    @staticmethod
    def add_tag(ticket_id: int, tag: Tag):
        with SessionLocal() as s:
            ticket = s.query(Ticket).get(ticket_id)
            ticket.tags.append(tag)
            s.commit()

    @staticmethod
    def list_pool(user: User) -> List[Ticket]:
        with SessionLocal() as s:
            tickets = (
                s.query(Ticket)
                .filter(Ticket.status == "open", Ticket.min_role <= user.role)
                .all()
            )
            return tickets

    @staticmethod
    def solve(ticket_id: int, solution: str):
        with SessionLocal() as s:
            ticket = s.query(Ticket).get(ticket_id)
            ticket.solution = solution
            ticket.status = "closed"
            s.commit()

    @staticmethod
    def export_to_md(path: str):
        with SessionLocal() as s:
            tickets = s.query(Ticket).all()
            with open(path, "w", encoding="utf-8") as f:
                for t in tickets:
                    tags_line = ", ".join(tag.name for tag in t.tags)
                    f.write(f"### Ticket #{t.id}\n")
                    f.write(f"Role: {Role(t.min_role).name}\n\n")
                    f.write("Description:\n")
                    f.write(t.description.strip() + "\n\n")
                    if t.solution:
                        f.write("Solution:\n")
                        f.write(t.solution.strip() + "\n\n")
                    if tags_line:
                        f.write(f"Tags: {tags_line}\n\n")
                    f.write("---\n\n")

class RagService:
    def __init__(self):
        load_dotenv()
        self.sdk = YCloudML(
            folder_id=os.getenv("YANDEX_FOLDER_ID"),
            auth=os.getenv("YANDEX_API_KEY"),
        )

    def build_agent(self, md_path: str):
        file_obj = self.sdk.files.upload(md_path)
        op = self.sdk.search_indexes.create_deferred(
            [file_obj], index_type=TextSearchIndexType()
        )
        text_index = op.wait()
        text_tool = self.sdk.tools.search_index(text_index)
        model = self.sdk.models.completions("yandexgpt", model_version="rc")
        assistant = self.sdk.assistants.create(model, tools=[text_tool])
        return assistant

    def ask(self, question: str, md_path: str) -> str:
        assistant = self.build_agent(md_path)
        thread = self.sdk.threads.create()
        thread.write(question)
        run = assistant.run(thread)
        result = run.wait().message
        parts = [p.text for p in result.parts if hasattr(p, "text")]
        return "\n".join(parts)

rag_service = RagService()

bot = telebot.TeleBot(os.getenv("BOT_TOKEN"), parse_mode="HTML")
user_sessions: Dict[int, Dict] = {}

def main_keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("Создать тикет", "Создать тег")
    kb.add("Задать роль", "Мой пул", "Спросить у агента")
    return kb

@bot.message_handler(commands=["start"])
def start_cmd(message):
    UserService.get(message.from_user.id)
    bot.send_message(message.chat.id, "Меню", reply_markup=main_keyboard())

@bot.message_handler(func=lambda m: m.text == "Создать тег")
def tag_create_step(message):
    user_sessions[message.from_user.id] = {"state": "await_new_tag"}
    bot.send_message(message.chat.id, "Введите имя нового тега")

@bot.message_handler(func=lambda m: user_sessions.get(m.from_user.id, {}).get("state") == "await_new_tag")
def tag_create_finish(message):
    TagService.create(message.text.strip())
    bot.send_message(message.chat.id, "Тег создан")
    user_sessions.pop(message.from_user.id, None)

@bot.message_handler(func=lambda m: m.text == "Создать тикет")
def create_ticket_desc(message):
    user_sessions[message.from_user.id] = {"state": "await_desc"}
    bot.send_message(message.chat.id, "Опишите проблему")

@bot.message_handler(func=lambda m: user_sessions.get(m.from_user.id, {}).get("state") == "await_desc")
def create_ticket_role_select(message):
    session = user_sessions[message.from_user.id]
    session["description"] = message.text
    kb = types.InlineKeyboardMarkup()
    for r in Role:
        kb.add(types.InlineKeyboardButton(r.name, callback_data=f"ticket_role:{int(r)}"))
    bot.send_message(message.chat.id, "Выберите минимальную роль, которой доступна задача", reply_markup=kb)

@bot.callback_query_handler(func=lambda c: c.data.startswith("ticket_role:"))
def create_ticket_finalize(call):
    role_val = int(call.data.split(":")[1])
    session = user_sessions.pop(call.from_user.id)
    user = UserService.get(call.from_user.id)
    ticket = TicketService.create(user, session["description"], Role(role_val))
    ikb = types.InlineKeyboardMarkup()
    ikb.add(types.InlineKeyboardButton("Добавить решение", callback_data=f"add_solution:{ticket.id}"))
    ikb.add(types.InlineKeyboardButton("Добавить теги", callback_data=f"add_tags:{ticket.id}"))
    bot.send_message(call.message.chat.id, f"Тикет #{ticket.id} создан", reply_markup=ikb)

@bot.callback_query_handler(func=lambda c: c.data.startswith("add_solution:"))
def add_solution_start(call):
    ticket_id = int(call.data.split(":")[1])
    user_sessions[call.from_user.id] = {"state": "await_solution", "ticket_id": ticket_id}
    bot.send_message(call.message.chat.id, "Введите решение")

@bot.message_handler(func=lambda m: user_sessions.get(m.from_user.id, {}).get("state") == "await_solution")
def add_solution_finish(message):
    session = user_sessions.pop(message.from_user.id)
    TicketService.solve(session["ticket_id"], message.text)
    bot.send_message(message.chat.id, "Решение сохранено")
    filename = f"knowledge.md"
    TicketService.export_to_md(filename)
    RAG(filename)
    

@bot.callback_query_handler(func=lambda c: c.data.startswith("add_tags:"))
def add_tags_start(call):
    ticket_id = int(call.data.split(":")[1])
    user_sessions[call.from_user.id] = {"state": "adding_tags", "ticket_id": ticket_id}
    show_tag_keyboard(call.message.chat.id, ticket_id)

def show_tag_keyboard(chat_id: int, ticket_id: int):
    tags = TagService.all()
    kb = types.InlineKeyboardMarkup(row_width=2)
    for tag in tags:
        kb.add(types.InlineKeyboardButton(tag.name, callback_data=f"tag_select:{ticket_id}:{tag.id}"))
    kb.add(types.InlineKeyboardButton("Готово", callback_data="tag_done"))
    bot.send_message(chat_id, "Выберите тег", reply_markup=kb)

@bot.callback_query_handler(func=lambda c: c.data.startswith("tag_select:"))
def tag_select(call):
    _, ticket_id, tag_id = call.data.split(":")
    tag = TagService.get(TagService.get(int(tag_id)).name)
    TicketService.add_tag(int(ticket_id), tag)
    bot.answer_callback_query(call.id, "Тег добавлен")

@bot.callback_query_handler(func=lambda c: c.data == "tag_done")
def tag_done(call):
    bot.send_message(call.message.chat.id, "Теги добавлены")

@bot.message_handler(func=lambda m: m.text == "Задать роль")
def set_role_start(message):
    kb = types.InlineKeyboardMarkup()
    for r in Role:
        kb.add(types.InlineKeyboardButton(r.name, callback_data=f"user_role:{int(r)}"))
    bot.send_message(message.chat.id, "Выберите роль", reply_markup=kb)

@bot.callback_query_handler(func=lambda c: c.data.startswith("user_role:"))
def set_role_finish(call):
    role_val = int(call.data.split(":")[1])
    user = UserService.get(call.from_user.id)
    UserService.set_role(user, Role(role_val))
    bot.send_message(call.message.chat.id, "Роль сохранена")

@bot.message_handler(func=lambda m: m.text == "Мой пул")
def my_pool(message):
    user = UserService.get(message.from_user.id)
    tickets = TicketService.list_pool(user)
    if not tickets:
        bot.send_message(message.chat.id, "Нет задач")
        return
    for t in tickets:
        ikb = types.InlineKeyboardMarkup()
        ikb.add(types.InlineKeyboardButton("Решить", callback_data=f"add_solution:{t.id}"))
        desc = t.description[:100]
        bot.send_message(message.chat.id, f"#{t.id} {desc}", reply_markup=ikb)

@bot.message_handler(func=lambda m: m.text == "Спросить у агента")
def ask_agent_start(message):
    user_sessions[message.from_user.id] = {"state": "await_query"}
    bot.send_message(message.chat.id, "Введите вопрос")


import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    TextSearchIndexType
)
load_dotenv()
def RAG(filename):
    sdk = YCloudML(
        folder_id=os.getenv("YANDEX_FOLDER_ID"),
        auth=os.getenv("YANDEX_API_KEY"),
    )

    file = sdk.files.upload(filename)
    operation = sdk.search_indexes.create_deferred([file], index_type=TextSearchIndexType())
    text_index = operation.wait()

    text_tool = sdk.tools.search_index(text_index)
    model = sdk.models.completions("yandexgpt", model_version="rc")
    assistant = sdk.assistants.create(model, tools=[text_tool])

    print(assistant.id)
    with open('assistant_id.txt', 'w', encoding='utf-8') as f:
        f.write(assistant.id)

def llm(query):
    sdk = YCloudML(
        folder_id=os.getenv("YANDEX_FOLDER_ID"),
        auth=os.getenv("YANDEX_API_KEY"),
    )
    assistant = sdk.assistants.get(open('assistant_id.txt').read())

    text_index_thread = sdk.threads.create()
    text_index_thread.write(query)
    run = assistant.run(text_index_thread)

    result = run.wait().message
    return '\n'.join(result.parts)

@bot.message_handler(func=lambda m: user_sessions.get(m.from_user.id, {}).get("state") == "await_query")
def ask_agent_process(message):
    query = message.text
    user_sessions.pop(message.from_user.id, None)
    answer = llm(query)
    if answer.strip():
        bot.send_message(message.chat.id, answer)
    else:
        bot.send_message(message.chat.id, "Нет ответа от модели")

if __name__ == "__main__":
    init_db()
    bot.infinity_polling()
