============================================================
ERROR ANALYSIS: SWITCH-POINT TOKENS (HELD-OUT TEST SET)
============================================================

Total switch-point token positions evaluated: 596
Accuracy at switch points: 0.4027 (240/596)

------------------------------------------------------------
TOP 15 MISMATCH PAIRS AT SWITCH POINTS (gold → predicted)
------------------------------------------------------------
Gold         Predicted     Count  % of SP errors
NOUN         PROPN            48           13.5%
SCONJ        ADV              35            9.8%
PUNCT        X                24            6.7%
PUNCT        INTJ             20            5.6%
NOUN         VERB             19            5.3%
VERB         NOUN             18            5.1%
ADJ          PROPN            14            3.9%
VERB         PROPN            13            3.7%
AUX          VERB             13            3.7%
ADJ          NOUN             11            3.1%
ADJ          ADV              11            3.1%
PROPN        NOUN             10            2.8%
VERB         ADP               7            2.0%
ADV          NOUN              7            2.0%
PRON         DET               6            1.7%

------------------------------------------------------------
TOP 5 ERROR PAIRS — EXAMPLE SENTENCES
------------------------------------------------------------

#1. Gold=NOUN → Predicted=PROPN  (n=48)
  Example 1:
    Context:    这 | 个 | [railway] | 上 | 玩
    Token:      railway  (language: EN)
    Predicted:  PROPN
    Gold:       NOUN
  Example 2:
    Context:    这 | 个 | [rule] | 和 | save
    Token:      rule  (language: EN)
    Predicted:  PROPN
    Gold:       NOUN
  Example 3:
    Context:    这 | 个 | [system]
    Token:      system  (language: EN)
    Predicted:  PROPN
    Gold:       NOUN

#2. Gold=SCONJ → Predicted=ADV  (n=35)
  Example 1:
    Context:    something | correct | [但] | 是 | 你
    Token:      但  (language: ZH)
    Predicted:  ADV
    Gold:       SCONJ
  Example 2:
    Context:    些 | judging | [就] | 需 | 要
    Token:      就  (language: ZH)
    Predicted:  ADV
    Gold:       SCONJ
  Example 3:
    Context:    something | correct | [但] | 是 | 那
    Token:      但  (language: ZH)
    Predicted:  ADV
    Gold:       SCONJ

#3. Gold=PUNCT → Predicted=X  (n=24)
  Example 1:
    Context:    呃 | 去 | [[] | UNK | ]
    Token:      [  (language: EN)
    Predicted:  X
    Gold:       PUNCT
  Example 2:
    Context:    如 | 果 | [[] | UNK | ]
    Token:      [  (language: EN)
    Predicted:  X
    Gold:       PUNCT
  Example 3:
    Context:    学 | 校 | [[] | UNK | ]
    Token:      [  (language: EN)
    Predicted:  X
    Gold:       PUNCT

#4. Gold=PUNCT → Predicted=INTJ  (n=20)
  Example 1:
    Context:    UNK | ] | [呃] | 如 | 果
    Token:      呃  (language: ZH)
    Predicted:  INTJ
    Gold:       PUNCT
  Example 2:
    Context:    呀 | or | [嗯]
    Token:      嗯  (language: ZH)
    Predicted:  INTJ
    Gold:       PUNCT
  Example 3:
    Context:    yeah | because | [呃] | 我 | 觉
    Token:      呃  (language: ZH)
    Predicted:  INTJ
    Gold:       PUNCT

#5. Gold=NOUN → Predicted=VERB  (n=19)
  Example 1:
    Context:    question | [会] | 比 | 较
    Token:      会  (language: ZH)
    Predicted:  VERB
    Gold:       NOUN
  Example 2:
    Context:    个 | 人 | [update] | 他 | 的
    Token:      update  (language: EN)
    Predicted:  VERB
    Gold:       NOUN
  Example 3:
    Context:    useless | 的 | [information] | 你 | 觉
    Token:      information  (language: EN)
    Predicted:  VERB
    Gold:       NOUN

------------------------------------------------------------
AT SWITCH POINT vs. ONE POSITION AFTER
------------------------------------------------------------
  Tokens AT switch point:         accuracy = 0.4027  (240/596)
  Tokens ONE AFTER switch point:  accuracy = 0.5164  (205/397)
  Difference: 0.1137 (AT harder)

------------------------------------------------------------
ERROR BREAKDOWN BY LANGUAGE LABEL AT SWITCH POINTS
------------------------------------------------------------
  EN-labeled switch-point tokens: accuracy = 0.3493  (117/335)
  ZH-labeled switch-point tokens: accuracy = 0.4713  (123/261)