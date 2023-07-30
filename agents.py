from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from tools import get_tools

class Agent:
    agent = None
    llm = None

    # initialise the agent
    def __init__(self):
        self.empty()
        self.llm = ChatOpenAI()

        # initialise agent execute
        self.agent = initialize_agent(
            get_tools(),
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True),
            handle_parsing_errors="Check the output and correct it to make it conform.",
            verbose=True,
            )

    def run(self, data):
        return self.agent.run(data)

    def empty(self):
        self.agent = None
        self.llm = None

    def reset(self):
        print("\033[96mReset agent has been triggered\033[0m")
        self.__init__()
