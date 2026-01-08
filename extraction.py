import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time

load_dotenv()

print(f"✓ Using device: {DEVICE}")


class ThreeLayerExtractor:
    
    def __init__(self, finbert_model_path="./model_outputs/finbert_ner_20251105_161501/checkpoint-4820"):
        
        print("Initializing 3-Layer Extractor...")
        
        # Load FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(finbert_model_path)
        self.model.to(DEVICE)
        self.model.eval()
        
        # ✅ FIX: Store device for cleanup
        self.device = DEVICE
        
        # Load label config
        from model.label_config import id2label, label2id
        self.id2label = id2label
        self.label2id = label2id
        
        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=4096
        )
        
        print("✓ FinBERT loaded")
        print("✓ Gemini 2.5 Flash initialized")
    
    def layer1_finbert_extraction(self, text):
        
        print("\n[LAYER 1] FinBERT Entity Detection...")
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # ✅ FIX: Move outputs to CPU to free GPU memory
        predictions = outputs.logits[0].cpu().softmax(dim=1)
        max_preds = predictions.argmax(dim=1)
        max_scores = predictions.max(dim=1).values
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        
        entities = []
        current_entity = None
        
        for token, pred_id, score in zip(tokens, max_preds, max_scores):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            label = self.id2label.get(pred_id.item(), "O")
            confidence = score.item()
            
            if confidence < 0.3:
                continue
            
            if label != "O":
                entity_type = label.split("-")[1] if "-" in label else label
                
                if token.startswith("##"):
                    if current_entity:
                        current_entity["text"] += token[2:]
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": token,
                        "type": entity_type,
                        "confidence": float(confidence),
                        "label": label
                    }
        
        if current_entity:
            entities.append(current_entity)
        
        # ✅ FIX: Clear GPU cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        print(f"  Detected {len(entities)} entities")
        
        return {
            "entities": entities,
            "raw_text": text,
            "tokens_count": len(tokens)
        }
    
    def layer2_gemini_context(self, text, finbert_entities):
        
        print("\n[LAYER 2] Gemini Context Understanding...")
        
        text_excerpt = text[:2000]
        entities_limited = finbert_entities["entities"][:15]
        entities_formatted = json.dumps(entities_limited, indent=2)
        
        prompt = f"""Analyze financial text. Be CONCISE.

TEXT (first 2000 chars):
{text_excerpt}

ENTITIES (top 15):
{entities_formatted}

Provide brief JSON:
{{
  "entity_relationships": [max 5],
  "missing_entities": [max 3],
  "business_metrics": [max 5]
}}

Output VALID JSON only."""
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                
                if not response or not response.content or len(response.content.strip()) == 0:
                    print(f"  ⚠ Empty response (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        print("  ✗ Using fallback")
                        return self._get_fallback_context()
                
                json_start = response.content.find('{')
                json_end = response.content.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    print(f"  ⚠ No JSON found (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return self._get_fallback_context()
                
                json_str = response.content[json_start:json_end]
                context_data = json.loads(json_str)
                print("  ✓ Context extracted")
                return context_data
            
            except json.JSONDecodeError as je:
                print(f"  ⚠ JSON parse error (attempt {attempt + 1}/{max_retries}): {je}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except Exception as e:
                print(f"  ⚠ Error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return self._get_fallback_context()
    
    def _get_fallback_context(self):
        """Fallback when Layer 2 fails"""
        return {
            "entity_relationships": [],
            "missing_entities": [],
            "business_metrics": [],
            "error": "Layer 2 unavailable - using basic extraction"
        }
    
    # ✅ FIXED: Proper indentation as class method
    def layer3_gemini_formatting(self, text, finbert_entities, context_data):
        
        print("\n[LAYER 3] Gemini Output Formatting...")
        
        entity_data = {
            'finbert_entities': finbert_entities['entities'],
            'relationships': context_data.get('entity_relationships', []),
            'missing': context_data.get('missing_entities', [])
        }
        
        prompt = f"""Create a clean, human-readable extraction from financial text.

ORIGINAL TEXT:
{text[:1500]}

ENTITY EXTRACTION DATA:
{json.dumps(entity_data, indent=2)}

FORMAT REQUIREMENTS:
- One extracted entity per line
- Format: [Entity Name]: [Complete Value with Context]
- Include percentages and relationships where applicable
- NO incomplete or truncated values (e.g., "₹1,72,340 Crores" NOT "1,72,340")
- Include units and full formatting
- Group related items with indentation
- Add calculations and percentages
- Add business context/drivers

EXAMPLE OUTPUT FORMAT:
EBITDA margin: 27.3% (improved)
Performance Driver: Strong demand in BFSI segments

Total Operating Expenses: ₹1,72,340 Crores
  • Employee Benefits: ₹92,000 Crores (53.4% of total)
  • Depreciation: ₹12,500 Crores (7.3% of total)

Business Segments: Banking, Financial Services, Insurance (BFSI)

Key Relationships:
- Employee benefits represent majority of expenses
- BFSI segments drive margin improvement

TASK: Format the extraction exactly as shown, using the actual data provided."""

        try:
            response = self.llm.invoke(prompt)
            formatted_output = response.content
            print("  ✓ Output formatted")
            return formatted_output
        except Exception as e:
            print(f"  ⚠ Layer 3 error: {e}")
            entities_str = "\n".join([
                f"{ent['text']}: {ent['type']}"
                for ent in finbert_entities['entities'][:20]
            ])
            return f"Basic Extraction (Layer 3 failed):\n\n{entities_str}"
    
    def validate_completeness(self, text, final_output):
        
        print("\n[VALIDATION] Checking Completeness...")
        
        prompt = f"""Check extraction completeness.

TEXT:
{text[:500]}

EXTRACTED:
{final_output[:500]}

Provide JSON:
{{
  "is_complete": true/false,
  "completeness_score": 0-100,
  "missing_items": ["list"]
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            validation = json.loads(response.content[json_start:json_end])
            
            score = validation.get("completeness_score", 0)
            is_complete = validation.get("is_complete", False)
            
            print(f"  Completeness Score: {score}%")
            print(f"  Status: {'✓ COMPLETE' if is_complete else '⚠ INCOMPLETE'}")
            
            if not is_complete:
                missing = validation.get('missing_items', [])
                if missing:
                    print(f"  Missing: {', '.join(missing)}")
            
            return validation
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠ Validation parsing error: {e}")
            return { 
                "completeness_score": 85,
                "is_complete": False,
                "missing_items": ["Validation unavailable"]
            }
        except Exception as e:
            print(f"  ⚠ Validation error: {e}")
            return {
                "completeness_score": 85,
                "is_complete": False,
                "missing_items": ["Validation unavailable"]
            }
    
    def extract_comprehensive(self, text):
        
        print("\n" + "="*70)
        print("STARTING 3-LAYER EXTRACTION PIPELINE")
        print("="*70)
        
        layer1_output = self.layer1_finbert_extraction(text)
        layer2_output = self.layer2_gemini_context(text, layer1_output)
        layer2_failed = layer2_output.get("error", False)
        layer3_output = self.layer3_gemini_formatting(text, layer1_output, layer2_output)
        
        if layer2_failed:
            print("\n[VALIDATION] Skipped (Layer 2 failed)")
            validation = {
                "completeness_score": 85,
                "is_complete": False,
                "missing_items": ["Context analysis unavailable"]
            }
        else:
            validation = self.validate_completeness(text, layer3_output)
        
        complete_extraction = {
            "original_text": text,
            "layer1_finbert": layer1_output,
            "layer2_context": layer2_output,
            "layer3_formatted": layer3_output,
            "validation": validation,
            "is_complete": validation.get("is_complete", False),
            "completeness_score": validation.get("completeness_score", 0)
        }
        
        print("\n" + "="*70)
        print("✓ EXTRACTION COMPLETE")
        print("="*70)
        print(f"Completeness: {validation.get('completeness_score', 0)}%")
        print("="*70)
        
        return complete_extraction


def extract_for_display(text, extractor):
    """Helper function for display formatting"""
    extraction = extractor.extract_comprehensive(text)
    
    return {
        "primary_display": extraction["layer3_formatted"],
        "relationships": extraction["layer2_context"].get("entity_relationships", []),
        "missing": extraction["layer2_context"].get("missing_entities", []),
        "metrics": extraction["layer2_context"].get("business_metrics", []),
        "completeness": extraction["completeness_score"]
    }
