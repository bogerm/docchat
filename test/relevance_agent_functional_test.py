from agents.relevance_checker import RelevanceChecker
from langchain_core.documents import Document

print('Testing complete relevance checker flow...')
print('=' * 60)

# Test case 1: Can answer
docs1 = [Document(page_content='Python is a high-level programming language known for its simplicity and readability.')]
# Test case 2: Partial match  
docs2 = [Document(page_content='Programming languages are tools used by developers.')]
# Test case 3: No match
docs3 = [Document(page_content='The weather today is sunny and warm.')]

class MockRetriever:
    def __init__(self, docs):
        self.docs = docs
    def invoke(self, query):
        return self.docs

checker = RelevanceChecker()

print('Test 1 - Should be CAN_ANSWER or PARTIAL:')
r1 = checker.check('What is Python?', MockRetriever(docs1), k=1)
print(f'  Result: {r1}')

print('\\nTest 2 - Should be PARTIAL:')
r2 = checker.check('What is Python?', MockRetriever(docs2), k=1)
print(f'  Result: {r2}')

print('\\nTest 3 - Should be NO_MATCH:')
r3 = checker.check('What is Python?', MockRetriever(docs3), k=1)
print(f'  Result: {r3}')

print('\\n' + '=' * 60)
print('✅ All tests completed!')
valid = all(r in {'CAN_ANSWER', 'PARTIAL', 'NO_MATCH'} for r in [r1, r2, r3])
print(f'All responses valid: {valid}')
