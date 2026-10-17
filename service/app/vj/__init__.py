"""VideoJev-4B: локальная видеомодель с прямым скорингом вариантов (System One).

runtime.py — загрузка модели под T4 (FP16, патч attention), окна, скоринг вопросов.
queries.py — вопросы к модели по каждому типу нарушения из rules.CATALOG.
detector.py — Detector для analyzer: окна → гейт по порогам → таймкод по grounding.
"""
