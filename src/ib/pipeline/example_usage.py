"""
Example usage of dataset-specific compression protocols.

This demonstrates how to use the specialized compression protocols
for different datasets (Wildchat, Fineweb) with their specific
compression strategies and prompts.
"""

from src.ib.clients import ClientConfig
from src.ib.pipeline import create_compression_protocol
from src.ib.pipeline.wildchat_compression import WildchatCompressionProtocol
from src.ib.pipeline.fineweb_compression import FinewebCompressionProtocol
from src.ib.tasks.base import Document


def example_wildchat_compression():
    """Example of using WildchatCompressionProtocol for conversation data."""
    
    # Configuration for Wildchat-specific compression
    config = WildchatCompressionProtocol.Config(
        predictor_client=ClientConfig(
            model_name="gpt-4o",
            api_key="your-openai-key"
        ),
        compressor_client=ClientConfig(
            model_name="llama-3.1-8b-instruct",
            api_base_url="your-sglang-url"
        ),
        # Wildchat-specific settings
        general_compression=True,  # Use general vs question-specific compression
        per_conversation_compression=True,  # Compress each conversation separately
        memory_type_name="chat memory",
        num_samples=3,
        worker_temperature=0.6,
    )
    
    # Create protocol instance
    protocol = WildchatCompressionProtocol(config)
    
    # Example conversation data
    context = [
        Document(
            title="User Conversations",
            content="""
            CHAT 1:
            User: I'm planning a trip to Japan next month.
            Assistant: That sounds exciting! What cities are you planning to visit?
            User: Tokyo and Kyoto mainly.
            Assistant: Great choices! Tokyo offers modern city life while Kyoto has beautiful temples.
            
            CHAT 2:
            User: What's the weather like in Japan in March?
            Assistant: March is a lovely time to visit Japan. It's spring with mild temperatures around 10-15°C.
            User: Perfect! Should I pack a jacket?
            Assistant: Yes, I'd recommend a light jacket for evenings and potential rain.
            """
        )
    ]
    
    # Query that builds on previous conversations (memory-based)
    query = "Based on our previous discussions, what specific items should I pack for my Japan trip?"
    
    # Run compression and prediction
    response = protocol(
        id="wildchat_example_1",
        query=query,
        context=context,
        num_samples=1
    )
    
    print(f"Wildchat Response: {response.text}")
    return response


def example_fineweb_compression():
    """Example of using FinewebCompressionProtocol for document data."""
    
    # Configuration for Fineweb-specific compression
    config = FinewebCompressionProtocol.Config(
        predictor_client=ClientConfig(
            model_name="gpt-4o",
            api_key="your-openai-key"
        ),
        compressor_client=ClientConfig(
            model_name="llama-3.1-8b-instruct", 
            api_base_url="your-sglang-url"
        ),
        # Fineweb-specific settings
        num_samples=2,
        max_sentences=3,  # Limit summary length
    )
    
    # Create protocol instance
    protocol = FinewebCompressionProtocol(config)
    
    # Example document data
    context = [
        Document(
            title="Climate Change Article",
            content="""
            Climate change refers to long-term shifts in global temperatures and weather patterns.
            While climate variations are natural, scientific evidence shows that human activities 
            have been the primary driver of climate change since the 1950s.
            
            The burning of fossil fuels generates greenhouse gas emissions that act like a blanket
            wrapped around the Earth, trapping the sun's heat and raising temperatures. The main
            greenhouse gases include carbon dioxide and methane.
            
            Consequences of climate change include rising sea levels, extreme weather events,
            droughts, flooding, and threats to food security. These impacts affect communities
            worldwide, with vulnerable populations facing the greatest risks.
            
            Solutions involve both mitigation (reducing emissions) and adaptation (adjusting to
            climate impacts). Renewable energy, energy efficiency, and sustainable transportation
            are key mitigation strategies.
            """
        )
    ]
    
    # Query about the document content
    query = "What are the main causes and effects of climate change mentioned in the article?"
    
    # Run compression and prediction
    response = protocol(
        id="fineweb_example_1", 
        query=query,
        context=context,
        num_samples=1
    )
    
    print(f"Fineweb Response: {response.text}")
    return response


def example_factory_usage():
    """Example of using the factory function to create protocols."""
    
    # Base configuration
    base_config = {
        "predictor_client": ClientConfig(
            model_name="gpt-4o",
            api_key="your-openai-key"
        ),
        "compressor_client": ClientConfig(
            model_name="llama-3.1-8b-instruct",
            api_base_url="your-sglang-url"
        ),
        "num_samples": 1,
    }
    
    # Create Wildchat protocol using factory
    wildchat_config = WildchatCompressionProtocol.Config(
        **base_config,
        dataset_type="wildchat",
        general_compression=False,  # Use question-specific compression
    )
    wildchat_protocol = create_compression_protocol(wildchat_config)
    print(f"Created protocol: {type(wildchat_protocol).__name__}")
    
    # Create Fineweb protocol using factory  
    fineweb_config = FinewebCompressionProtocol.Config(
        **base_config,
        dataset_type="fineweb",
        max_sentences=5,
    )
    fineweb_protocol = create_compression_protocol(fineweb_config)
    print(f"Created protocol: {type(fineweb_protocol).__name__}")
    
    return wildchat_protocol, fineweb_protocol


if __name__ == "__main__":
    print("=== Dataset-Specific Compression Protocol Examples ===\n")
    
    print("1. Factory Pattern Usage:")
    example_factory_usage()
    print()
    
    print("2. Wildchat Compression Example:")
    # example_wildchat_compression()
    print("(Uncomment to run with actual API keys)")
    print()
    
    print("3. Fineweb Compression Example:")  
    # example_fineweb_compression()
    print("(Uncomment to run with actual API keys)")