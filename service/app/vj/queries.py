"""Вопросы к VideoJev по типам нарушений.

Модель обучена на английских данных, поэтому вопросы на английском. Каждый тип задаётся:
- gate: вопрос с вариантами; вероятность варианта yes_index — оценка «признак виден»;
- ground: описание события для temporal grounding (где именно оно происходит);
- state: True, если нарушение — длящееся состояние (сейф/дверь открыты). Тогда таймкод —
  начало отрезка, а не пик.

Формулировки описывают только видимое действие, без юридической оценки.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TypeQuery:
    code: str  # ключ rules.CATALOG
    gate: str
    options: tuple[str, ...]
    yes_index: int
    ground: str
    state: bool = False


# В каждом гейте вариант A — признак нарушения, B — как выглядит норма, C — ситуация не возникла.
# Явная альтернатива «как правильно» заметно снижает ложные срабатывания по сравнению с «Да/Нет».
QUERIES: tuple[TypeQuery, ...] = (
    TypeQuery(
        code="SAFE_OPEN",
        gate="What happens with the metal safe in this video?",
        options=("A. The safe door stays open after the cashier takes things from it.",
                 "B. The cashier closes the safe after using it.",
                 "C. The safe is not opened."),
        yes_index=0,
        ground="The door of the safe is left open while the cashier is not using the safe.",
        state=True,
    ),
    TypeQuery(
        code="UNATTENDED_CASH",
        gate="What happens to the money when the cashier leaves the desk?",
        options=("A. The cashier walks away while banknotes or a cash bag remain on the desk.",
                 "B. The cashier puts the money away before leaving the desk.",
                 "C. The cashier does not leave the desk."),
        yes_index=0,
        ground="The cashier walks away and leaves money or a bag on the desk.",
    ),
    TypeQuery(
        code="DOOR_OPEN",
        gate="What happens with the door of the cash room?",
        options=("A. A person enters or leaves and the door is left open.",
                 "B. The door is closed after every entry and exit.",
                 "C. Nobody uses the door."),
        yes_index=0,
        ground="A person passes through the door and the door is left open.",
        state=True,
    ),
    TypeQuery(
        code="NO_COUNTER",
        gate="How does the cashier count the banknotes?",
        options=("A. By hand, without using the banknote counting machine.",
                 "B. With the banknote counting machine.",
                 "C. No banknotes are counted."),
        yes_index=0,
        ground="The cashier counts banknotes by hand without the counting machine.",
    ),
    TypeQuery(
        code="UNAUTHORIZED_PERSON",
        gate="Where do other people stay while the cashier handles cash?",
        options=("A. Another person comes behind the desk or stands right next to the cashier.",
                 "B. Other people stay on the client side of the desk.",
                 "C. Nobody else is in the room."),
        yes_index=0,
        ground="A second person comes behind the desk and stands next to the cashier.",
    ),
    TypeQuery(
        code="CASH_EXPOSED_TO_VISITOR",
        gate="Where is the cash when a new visitor enters the room?",
        options=("A. Banknotes lie openly on the desk when the visitor enters.",
                 "B. The desk is clear of cash when the visitor enters.",
                 "C. No new visitor enters."),
        yes_index=0,
        ground="A visitor enters the room while banknotes lie on the desk.",
    ),
    TypeQuery(
        code="CIT_PROCEDURE",
        gate="How does the cashier receive a bag from a cash collector?",
        options=("A. The collector hands the bag across the desk and the cashier just takes it.",
                 "B. The cashier inspects the bag and its seal before accepting it.",
                 "C. No cash collector brings a bag."),
        yes_index=0,
        ground="A cash collector hands a bag to the cashier across the desk.",
    ),
    TypeQuery(
        code="PERSONAL_ITEMS",
        gate="What is on the cashier's desk besides work equipment and money?",
        options=("A. A personal bag, clothing, food, a drink or a smartphone.",
                 "B. Nothing personal, only work equipment and money."),
        yes_index=0,
        ground="A personal item, drink or phone is used at the cashier's desk.",
    ),
)

BY_CODE = {q.code: q for q in QUERIES}


def gate_questions() -> list[dict]:
    return [{"id": f"gate:{q.code}", "code": q.code, "task": "choice", "query": q.gate, "options": list(q.options)} for q in QUERIES]


def ground_questions(codes: list[str]) -> list[dict]:
    return [{"id": f"ground:{c}", "code": c, "task": "grounding", "query": BY_CODE[c].ground} for c in codes]
