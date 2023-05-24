from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

class QnAModel:
    def __init__(self):
        context = (
            """Black Knight (NYSE:BKI) is a leading provider of integrated software, data and analytics solutions that facilitate and automate many of the business processes across the homeownership life cycle. 
            Black Knight is committed to being a premier business partner that clients rely on to achieve their strategic goals, realize greater success and better serve their customers by delivering best-in-class software, services and insights with a relentless commitment to excellence, innovation, integrity and leadership.
            Its mission is to be the PREMIER PROVIDER of software, data and analytics, known for CLIENT FOCUS AND PRODUCT EXCELLENCE; and to deliver INNOVATIVE, seamlessly INTEGRATED solutions with URGENCY.
        Anthony Jabbour is responsible for providing leadership and direction to the company’s management and Board of Directors.
        Before being appointed Executive Chairman, Anthony served as Chairman and CEO of Black Knight, where he helped substantially increase organic growth, led the company to deliver numerous digital solutions and other innovative capabilities, and oversaw nine acquisitions to provide greater shareholder value and help transform the industries Black Knight serves.
        Daniel Sudhindaran works at beeline and he is a director of product. He is also interested in cricket. 
        """
        )
        model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
        tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
        self.nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
        self.context = context
