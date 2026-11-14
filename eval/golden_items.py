"""
Hand-written golden questions added for the v2 system (data only; eval/build_golden.py
turns them into eval/golden.jsonl and verifies every piece of evidence against the filings).

Half of each list goes to the `test` split (odd positions), which is never used to tune
rules, thresholds or expansion terms.
"""

# (ticker, metric in data/facts/facts.json, fiscal year or None for the filing's year, question)
NUMERIC = [
    ("AAPL", "revenue", None, "What were Apple's total net sales in its latest fiscal year?"),
    ("AAPL", "net_income", None, "How much profit did Apple make last fiscal year?"),
    ("AAPL", "rd_expense", None, "How much does Apple spend on R&D a year?"),
    ("AAPL", "eps_diluted", None, "What did Apple earn per diluted share?"),
    ("MSFT", "revenue", 2025, "What was Microsoft's total revenue for fiscal 2025?"),
    ("MSFT", "operating_income", None, "What was Microsoft's operating income?"),
    ("MSFT", "revenue", 2024, "How much revenue did Microsoft report in fiscal 2024?"),
    ("MSFT", "cash", None, "How much cash and cash equivalents does Microsoft hold?"),
    ("GOOGL", "revenue", 2025, "What was Alphabet's revenue in 2025?"),
    ("GOOGL", "net_income", None, "What was Alphabet's net income?"),
    ("GOOGL", "capex", None, "How much did Alphabet spend on capital expenditures?"),
    ("GOOGL", "rd_expense", 2024, "What did Alphabet spend on research and development in 2024?"),
    ("AMZN", "revenue", 2025, "What were Amazon's net sales in 2025?"),
    ("AMZN", "operating_income", None, "What was Amazon's operating income for the year?"),
    ("AMZN", "operating_cash_flow", None, "How much operating cash flow did Amazon generate?"),
    ("AMZN", "cash", None, "How much cash does Amazon have?"),
    ("META", "revenue", 2025, "How much revenue did Meta bring in during 2025?"),
    ("META", "net_income", 2025, "What was Meta's net income in 2025?"),
    ("META", "rd_expense", None, "What does Meta spend on research and development?"),
    ("META", "total_assets", None, "What are Meta's total assets?"),
    ("NVDA", "revenue", 2026, "What was NVIDIA's revenue in fiscal 2026?"),
    ("NVDA", "net_income", None, "How much did NVIDIA earn in net income?"),
    ("NVDA", "gross_profit", None, "What was NVIDIA's gross profit?"),
    ("NVDA", "revenue", 2025, "What was NVIDIA's revenue in fiscal 2025?"),
    ("TSLA", "revenue", 2025, "What were Tesla's total revenues in 2025?"),
    ("TSLA", "net_income", 2025, "What was Tesla's net income for 2025?"),
    ("TSLA", "rd_expense", None, "How much did Tesla spend on R&D?"),
    ("TSLA", "cash", None, "What is Tesla's cash position?"),
    ("JPM", "net_income", 2025, "What was JPMorgan Chase's net income in 2025?"),
    ("JPM", "total_assets", None, "How large is JPMorgan's balance sheet in total assets?"),
    ("JPM", "eps_diluted", None, "What were JPMorgan's diluted earnings per share?"),
    ("JNJ", "revenue", 2025, "What were Johnson & Johnson's sales in 2025?"),
    ("JNJ", "rd_expense", None, "How much does J&J invest in research and development?"),
    ("JNJ", "net_income", None, "What were Johnson & Johnson's net earnings?"),
    ("XOM", "net_income", 2025, "What was Exxon Mobil's net income in 2025?"),
    ("XOM", "total_assets", None, "What are Exxon Mobil's total assets?"),
    ("XOM", "cash", None, "How much cash does Exxon have on hand?"),
    ("AMZN", "employees", None, "How many people does Amazon employ?"),
]

# ((ticker, ticker), metric, question): the answer must state both figures
COMPARISON = [
    (("AAPL", "MSFT"), "rd_expense", "Who spends more on R&D, Apple or Microsoft?"),
    (("GOOGL", "META"), "revenue", "Compare Alphabet's and Meta's revenue."),
    (("AMZN", "MSFT"), "operating_income", "Did Amazon or Microsoft have higher operating income?"),
    (("NVDA", "TSLA"), "net_income", "Which earned more net income, NVIDIA or Tesla?"),
    (("JPM", "XOM"), "total_assets", "Compare the total assets of JPMorgan and Exxon Mobil."),
    (("AAPL", "GOOGL"), "cash", "Who has more cash, Apple or Alphabet?"),
    (("JNJ", "META"), "employees", "Does Johnson & Johnson or Meta have more employees?"),
    (("TSLA", "AMZN"), "revenue", "How do Tesla's revenues compare with Amazon's net sales?"),
]

# (metric, "max" | "min", question): no company named; the answer must name the winner
CROSS_COMPANY = [
    ("employees", "max", "Which company has the most employees?"),
    ("revenue", "max", "Which of the companies had the highest revenue?"),
    ("net_income", "max", "Which company earned the most net income?"),
    ("rd_expense", "max", "Which company spends the most on research and development?"),
    ("net_income", "min", "Which company had the lowest net income?"),
    ("total_assets", "max", "Which company has the largest total assets?"),
]

# (ticker, question, verbatim evidence, string the answer must contain)
LOOKUP = [
    ("NVDA", "How concentrated is NVIDIA's revenue among its biggest customers?",
     "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to another direct customer represented 14% of total revenue", "22%"),
    ("JNJ", "How many manufacturing plants do Johnson & Johnson's subsidiaries run?",
     "The Company’s subsidiaries operate 63 manufacturing facilities", "63"),
    ("MSFT", "Who does Microsoft's chief information security officer report to?",
     "Our CISO leads the strategy, engineering, and operations of cybersecurity across the company, and reports to the EVP of Cloud + AI.", "Cloud + AI"),
    ("META", "Who leads Meta's cybersecurity program?",
     "Our Chief Information Security Officer (CISO), Guy Rosen, leads our cybersecurity program", "Guy Rosen"),
    ("JPM", "Who does JPMorgan's CISO report to?",
     "The Global Chief Information Security Officer (“CISO”) reports to the Global Chief Information Officer", "Chief Information Officer"),
    ("META", "What are Meta's two business segments?",
     "We report our financial results for our two reportable segments: Family of Apps (FoA) and Reality Labs (RL).", "Reality Labs"),
    ("TSLA", "What business segments does Tesla report?",
     "We operate as two reportable segments: (i) automotive and (ii) energy generation and storage.", "energy generation and storage"),
    ("JNJ", "Which segments is Johnson & Johnson organized into?",
     "the Company is now organized into two reportable segments: Innovative Medicine and MedTech.", "MedTech"),
    ("MSFT", "What are the three segments Microsoft uses to report results?",
     "We operate our business and report our financial performance using three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.", "Intelligent Cloud"),
    ("JPM", "What are JPMorgan Chase's reportable business segments?",
     "the Firm has three reportable business segments – Consumer & Community Banking (“CCB”), Commercial & Investment Bank (“CIB”) and Asset & Wealth Management (“AWM”)", "Asset & Wealth Management"),
    ("GOOGL", "How does Alphabet break its business into segments?",
     "We report Google in two segments, Google Services and Google Cloud, and all non-Google businesses collectively as Other Bets.", "Other Bets"),
    ("AMZN", "Is Amazon's business seasonal?",
     "Our business is affected by seasonality, which historically has resulted in higher sales volume during our fourth quarter, which ends December 31.", "fourth quarter"),
    ("AAPL", "How big is Apple's share buyback program announced in May 2024?",
     "On May 2, 2024, the Company announced a program to repurchase up to $110 billion of the Company’s common stock.", "110 billion"),
    ("AAPL", "Where are Apple's products manufactured?",
     "A significant majority of the Company’s manufacturing is performed in whole or in part by outsourcing partners located primarily in China mainland, India, Japan, South Korea, Taiwan and Vietnam", "Vietnam"),
    ("XOM", "What brand names do Exxon Mobil's divisions and affiliates use?",
     "Exxon Mobil Corporation's divisions and affiliates have many names, including ExxonMobil, Exxon, Esso, Mobil or XTO.", "Esso"),
    ("GOOGL", "Does Alphabet plan to keep paying dividends?",
     "The company intends to pay quarterly cash dividends in the future, subject to review and approval by the company’s Board of Directors in its sole discretion.", "quarterly cash dividends"),
]

# (ticker, question, 10-K item codes that hold the answer, regex a relevant passage must match)
THEMATIC = [
    ("TSLA", "What are the main risks Tesla faces from competition?", ["1A"], r"compet"),
    ("AAPL", "What supply chain risks does Apple describe?", ["1A"], r"suppl|outsourc|manufactur"),
    ("META", "What privacy and data protection risks does Meta face?", ["1A"], r"privacy|data protection|data use"),
    ("NVDA", "How could export controls hurt NVIDIA's business?", ["1A"], r"export control|licens"),
    ("JPM", "What cybersecurity threats does JPMorgan describe as risks?", ["1A"], r"cyber"),
    ("XOM", "How might climate policy and regulation affect Exxon Mobil?", ["1A"], r"climate|greenhouse|emission"),
    ("AMZN", "What does Amazon say about the competition it faces?", ["1", "1A"], r"compet"),
    ("MSFT", "What risks does Microsoft see in its AI investments?", ["1A"], r"\bAI\b|artificial intelligence"),
    ("GOOGL", "What regulatory and antitrust risks does Alphabet highlight?", ["1A"], r"antitrust|competition law|regulat"),
    ("JNJ", "What litigation risks does Johnson & Johnson describe?", ["1A"], r"litigation|lawsuit|legal proceeding|claims"),
    ("AMZN", "What properties and facilities does Amazon operate?", ["2"], r"square f|facilit|leased|owned"),
    ("TSLA", "How does Tesla describe its Supercharger network?", ["1"], r"Supercharg"),
    ("JNJ", "How does Johnson & Johnson protect its products with patents?", ["1"], r"patent"),
    ("NVDA", "What does NVIDIA say about its liquidity and capital resources?", ["7"], r"liquidity|cash|capital"),
    ("MSFT", "How does Microsoft manage cybersecurity risk?", ["1C"], r"cyber|security"),
    ("META", "What legal proceedings is Meta involved in?", ["3"], r"."),
    ("AAPL", "How does Apple describe its exposure to foreign exchange risk?", ["7A"], r"foreign|currenc|exchange"),
    ("JPM", "How does JPMorgan describe its human capital and workforce?", ["1"], r"employee|workforce|human capital"),
]

# (ticker or None, question, why it cannot be answered from the corpus)
UNANSWERABLE_V2 = [
    (None, "What was Disney's theme park revenue last year?", "other_company"),
    (None, "How many employees does Toyota have?", "other_company"),
    (None, "What is Pfizer's R&D budget?", "other_company"),
    (None, "How much revenue did Samsung make from semiconductors?", "other_company"),
    ("TSLA", "What is Tesla's current market capitalization?", "market_data"),
    ("MSFT", "Is Microsoft stock a good buy right now?", "opinion"),
    ("AMZN", "What will Amazon's revenue be next year?", "forecast"),
    ("GOOGL", "What is Alphabet's revenue guidance for 2027?", "forecast"),
    ("NVDA", "What was NVIDIA's revenue in the second quarter of fiscal 2027?", "other_period"),
    ("AAPL", "What was Apple's net income in 2010?", "other_period"),
    ("JNJ", "What was Johnson & Johnson's revenue in fiscal 2018?", "other_period"),
    ("META", "How many Threads users did Meta report in its October 2026 earnings release?", "not_in_filing"),
    ("XOM", "What is the name of Exxon Mobil's CEO's spouse?", "not_in_filing"),
    ("JPM", "What is JPMorgan's customer satisfaction score from J.D. Power?", "not_in_filing"),
    (None, "Who won the 2025 World Series?", "off_topic"),
    (None, "Write a poem about quarterly earnings.", "off_topic"),
]
