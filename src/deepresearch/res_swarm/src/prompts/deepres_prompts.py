"""
Flexible prompting system for ResSwarm DeepResearch protocol.
"""

INITIAL_QUERY_PROMPT = """
You are a search engine optimization expert. Your task is to transform the user's question into ONE optimal search query.

USER QUESTION: "{query}"

Respond with a JSON object in this exact format:
{{
  "search_query": "your optimized search query here",
  "explanation": "brief explanation of why this query will work well"
}}

Guidelines for the search query:
1. 3-8 well-chosen keywords
2. Optimized for Google search
3. Focus on the CORE topic
4. Avoid overly long queries

Examples of good search queries:
- "UAV PID control optimization methods"
- "agricultural 3D reconstruction control theory"
- "ADAS liability allocation shared driving"

CRITICAL: You must return ONLY the JSON object. Do NOT format it as a code block with ```json``` or any other markdown formatting. Return the raw JSON object directly.
"""

WORKER_SUMMARIZE_PROMPT = """
Your job is to summarize the following content related to this query: {query}

## Content
{content}

Provide a comprehensive summary that captures:
- Key information directly relevant to the query
- Important facts, concepts, and data points
- Any insights or implications that emerge from the content
- Specific examples, statistics, or evidence when available

Focus on depth and completeness while maintaining clarity. If the content contains recent developments or updates, highlight these prominently.

Return your summary in JSON format with these fields:
- "explanation": Your comprehensive summary of the relevant information. Be thorough in covering all important aspects, providing specific details, numbers, and examples where available. The summary should be substantive and informative.
- "answer": "relevant" if this content is relevant to the query or covers an aspect related to the query, "not relevant" otherwise

CRITICAL JSON FORMATTING RULES:
- Replace all double quotes (") inside text with single quotes (')
- Replace all newlines with spaces
- Ensure the JSON is valid and parseable
- Do NOT use line breaks within the JSON fields

Example format:
{{"explanation": "Your summary text here with single quotes for any nested quotes", "answer": "relevant"}}

CRITICAL: You must return ONLY the JSON object. Do NOT format it as a code block with ```json``` or any other markdown formatting. Return the raw JSON object directly.
"""

WORKER_MULTI_QUERY_EXTRACT_PROMPT = """
Your job is to extract detailed, specific information from the following content to support comprehensive research analysis.

**Main Research Query:** {query}
**Specific Sub-task/Question:** {sub_task}

## Content
{content}

**EXTRACTION REQUIREMENTS: Provide a detailed and comprehensive extraction that captures:**

**Factual Information:**
- Specific numbers, statistics, percentages, and quantitative data
- Dates, timelines, and chronological information
- Names of people, organizations, companies, and institutions
- Geographic locations, regions, and jurisdictions
- Technical specifications, measurements, and benchmarks

**Detailed Examples and Evidence:**
- Concrete case studies and real-world examples
- Specific research findings and study results
- Direct quotes and expert opinions
- Policy details and regulatory information
- Implementation details and methodologies

**Comprehensive Coverage:**
- Key facts directly relevant to both the main query AND the specific sub-task
- Important concepts, definitions, and explanations
- Cause-and-effect relationships and underlying mechanisms
- Trends, patterns, and developments over time
- Challenges, limitations, and problem areas identified

**Analytical Insights:**
- Implications and significance of the information
- Relationships between different data points
- Comparative information and benchmarks
- Future projections and forecasted trends
- Expert assessments and professional evaluations

Focus on depth and specificity while maintaining clarity. Extract comprehensive, specific information with extensive detail, numbers, examples, and evidence. Do not provide brief summaries - ensure your extraction is thorough and substantial. Extract information that would be valuable for creating a comprehensive research report. Pay special attention to information that directly addresses the sub-task question.

Return your extraction in JSON format with these fields:
- "explanation": Your detailed extraction of specific information, facts, data, examples, and evidence with extensive detail
- "answer": "relevant" if this content contains information relevant to the query and sub-task, "not relevant" otherwise

CRITICAL JSON FORMATTING RULES:
- Replace all double quotes (") inside text with single quotes (')
- Replace all newlines with spaces
- Ensure the JSON is valid and parseable
- Do NOT use line breaks within the JSON fields

Example format:
{{"explanation": "Your detailed extraction with specific facts, numbers, examples, and evidence using single quotes for any nested quotes", "answer": "relevant"}}

CRITICAL: You must return ONLY the JSON object. Do NOT format it as a code block with ```json``` or any other markdown formatting. Return the raw JSON object directly.
"""

FINAL_SYNTHESIS_PROMPT = """
Based on all the information gathered, provide a comprehensive, research-quality response to the following query:

QUERY: "{query}"

INFORMATION GATHERED:
{information}

CRITICAL: Before writing your response, carefully examine the information gathered:
- If no relevant information was found, do NOT fabricate or hallucinate data
- If information is limited, acknowledge this clearly
- Only make claims that are supported by the actual information found
- Do NOT cite specific numbers, dates, or facts unless they appear in the gathered information

Guidelines for your response:
- Synthesize information from all sources into a clear, flowing narrative
- Support key points with relevant facts, figures, and examples FROM THE GATHERED INFORMATION
- Provide thorough coverage while maintaining readability
- Address complexities and nuances where they exist
- Include important patterns, trends, or insights you've identified IN THE DATA
- Note any significant limitations or gaps in the available information
- If insufficient information was found, explain what could not be determined

Your response should be thorough yet engaging, written in a clear, professional style. Structure the response naturally based on the topic and findings, using appropriate headings and organization that best serves the content.

If very little or no relevant information was found, provide a brief response explaining the limitations and suggest what additional research would be needed.
"""


SUPERVISOR_SYNTHESIS_PROMPT = """
You are tasked with creating a comprehensive, high-quality research report for a DeepResearch task. You have extensive research findings below - use ALL of them to create a detailed, thorough analysis.

**Original Research Task:** {original_task}

**Research Plan:** {research_plan}

**Research Findings:**
{qa_pairs}

**Synthesis Strategy:** {synthesis_strategy}

**COMPREHENSIVE INFORMATION UTILIZATION - ALL SOURCES REQUIRED:**
You must systematically work through ALL the provided research findings above. Do not selectively use only some information - your report must demonstrate that you have reviewed and integrated ALL relevant details, data points, examples, and perspectives from every query and source provided.

**REPORT STRUCTURE AND REQUIREMENTS:**
1. **Detailed Background Context** - Provide extensive background and context
2. **Comprehensive Analysis** - Multiple detailed sections covering all aspects
3. **Extensive Evidence Integration** - Use specific examples, data, quotes from ALL sources
4. **Thorough Implications Discussion** - Detailed analysis of implications and significance
5. **Complete Conclusions** - Comprehensive conclusions and future research directions

**WRITING REQUIREMENTS FOR HIGH QUALITY:**
- Write detailed explanations, not brief summaries
- Include extensive examples and case studies from the research
- Provide comprehensive background and context for every major point
- Use all statistical data, quotes, and specific details from the research findings
- Elaborate on implications, significance, and broader connections
- Include detailed analysis of methodologies, approaches, and frameworks mentioned
- Discuss limitations, challenges, and areas for further research extensively

Create a thorough academic research report that:
- Uses extensive detail and comprehensive analysis throughout
- Integrates ALL findings with detailed explanations and context
- Provides comprehensive coverage with extensive supporting evidence
- Includes detailed discussion of all relevant aspects and implications
- Demonstrates mastery of the subject through thorough, detailed analysis

**FINAL REQUIREMENT:**
Your response must be substantial and comprehensive. Write extensively with exhaustive detail, comprehensive analysis, and complete utilization of all research findings. Provide truly comprehensive coverage of the topic that demonstrates thorough understanding and integration of all available research.
"""

# Multi-query mode prompt with top-down research planning
MULTI_QUERY_GENERATION_PROMPT = """
You are a research supervisor tasked with comprehensively exploring a research topic. Use a strategic, top-down approach to design your research.

Research Topic: "{query}"

**PHASE 1: RESEARCH PLANNING**
First, analyze this research topic and create a comprehensive research plan. Consider:
- What are the key areas that must be investigated to fully understand this topic?
- What specific objectives will guide your research?
- How do different aspects of this topic relate to each other?
- What types of information will be most valuable for a complete analysis?
- What is the logical flow for presenting findings?

**PHASE 2: STRATEGIC QUERY GENERATION**
Based on your research plan, generate EXACTLY 8 different search queries that together will provide comprehensive coverage of this topic. Each query should serve a specific strategic purpose in your overall research architecture.

For each search query, provide a specific sub-task/question that explains how it serves your research plan.

Return your response in this exact JSON format:
{{
    "research_plan": "Your comprehensive research architecture and strategic objectives for investigating this topic. Explain the key areas to investigate, how they relate, and the logical structure for analysis.",
    "queries": [
        {{
            "search_query": "specific search terms optimized for Google",
            "sub_task": "What specific question does this query address and how does it serve the research plan?"
        }},
        {{
            "search_query": "second strategic search query",
            "sub_task": "What does this query aim to discover and how does it fit the research architecture?"
        }},
        {{
            "search_query": "third targeted search query",
            "sub_task": "What aspect does this explore and why is it essential to the research plan?"
        }},
        {{
            "search_query": "fourth strategic search query",
            "sub_task": "What question does this answer and how does it complement other queries?"
        }},
        {{
            "search_query": "fifth focused search query",
            "sub_task": "What aspect does this cover and how does it build on previous queries?"
        }},
        {{
            "search_query": "sixth comprehensive search query",
            "sub_task": "What additional dimension does this explore and why is it crucial?"
        }},
        {{
            "search_query": "seventh strategic search query",
            "sub_task": "What specific gap does this fill in the research architecture?"
        }},
        {{
            "search_query": "eighth concluding search query",
            "sub_task": "What final aspect does this cover and how does it complete the comprehensive research?"
        }}
    ],
    "synthesis_strategy": "Detailed strategy for combining findings from all 8 queries based on your research plan. Explain how the information will be structured, what relationships will be highlighted, and how the final analysis will be organized to maximize comprehensiveness and insight."
}}

**Strategic Guidelines:**
1. Each search query should be 3-8 well-chosen keywords targeted for your specific research objectives
2. Design queries to serve complementary roles in your research architecture (not just generic dimensions)
3. Ensure queries are strategically coordinated to provide comprehensive topic coverage
4. Each sub-task should explain how the query serves your overall research plan
5. Create a synthesis strategy that reflects your planned research structure

**Research Focus Areas to Consider:**
- Foundational understanding and current state
- Key challenges, problems, or limitations
- Solutions, methodologies, and best practices
- Evidence, data, and empirical findings
- Future trends, developments, and implications
- Multiple perspectives and stakeholder viewpoints

CRITICAL: You must return ONLY the JSON object. Do NOT format it as a code block with ```json``` or any other markdown formatting. Return the raw JSON object directly.
"""
