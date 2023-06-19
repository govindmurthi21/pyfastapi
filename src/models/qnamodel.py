from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

class QnAModel:
    def __init__(self):
        context = (
            """
            Modern wealth management has changed and the legacy systems haven't kept up. There is more tech than ever but it's not solving your needs. We're here to help with Summit's AdvisorOS™ - built to replace last generation's portfolio management systems with a platform that unlocks efficiency, growth, and deeper relationships with clients.
            Two decades ago I was fortunate enough to bring meaningful innovation to the wealth management space. At Summit, we seek to drive abundance for advisors and their clients to make a much bigger impact.
            - Reed Colley, CEO & Co-founder
            Unlock a deeper understanding of your client's needs and engagement through a modern, digital and mobile experience. We’ve designed Summit to humanize wealth management and help guide your clients from a scarcity to an abundance mindset by elevating what they care about most.
        """
        )
        model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
        tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
        self.nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
        self.context = context
