"""
Chatbot RAG avec support multi-formats : CSV, TXT, PDF (avec OCR)
Version complète pour rapports et documents complexes
"""

import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dotenv import load_dotenv

# Import PDF (installation : pip install PyPDF2)
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyPDF2 non installé. Support PDF désactivé.")
    print("   Pour activer : pip install PyPDF2")

# Import OCR pour images dans PDFs (optionnel)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    print("⚠️  OCR non installé. Extraction d'images désactivée.")
    print("   Pour activer : pip install pdf2image pytesseract")

load_dotenv()

class ChatbotRAG:
    def __init__(self, groq_api_key: str = None, csv_folder: str = "knowledge_base"):
        """
        Initialise le chatbot avec support multi-formats
        
        Args:
            groq_api_key: Clé API Groq
            csv_folder: Dossier contenant les fichiers de connaissances
        """
        if groq_api_key is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("❌ Clé API Groq manquante!")
        
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        
        print("📥 Chargement du modèle d'embeddings...")
        #self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.documents = []
        self.embeddings = None
        self.index = None
        self.csv_folder = csv_folder
        self.conversation_history = []
        
        # Configuration PDF
        self.pdf_chunk_size = 1000  # Caractères par chunk
        self.pdf_overlap = 200      # Chevauchement entre chunks
        
        print("✅ Chatbot initialisé avec succès!")
        if PDF_SUPPORT:
            print("   ✓ Support PDF activé")
        if OCR_SUPPORT:
            print("   ✓ Support OCR activé (extraction images)")
    
    def load_pdf_file(self, file_path: Path, category: str = None) -> List[Dict]:
        """
        Charge un fichier PDF et extrait le texte
        
        Args:
            file_path: Chemin du fichier PDF
            category: Catégorie du document
        
        Returns:
            Liste de documents
        """
        if not PDF_SUPPORT:
            print(f"   ⚠️  PDF ignoré (PyPDF2 non installé): {file_path.name}")
            return []
        
        docs = []
        
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Métadonnées du PDF
                total_pages = len(pdf_reader.pages)
                pdf_metadata = {
                    "source": file_path.name,
                    "type": "pdf",
                    "category": category or file_path.parent.name,
                    "total_pages": total_pages,
                    "file_path": str(file_path)
                }
                
                # Extraire le texte de chaque page
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        full_text += f"\n[Page {page_num}/{total_pages}]\n{page_text}\n"
                
                # Découper en chunks intelligents
                if full_text.strip():
                    chunks = self._split_text_into_chunks(
                        full_text, 
                        chunk_size=self.pdf_chunk_size,
                        overlap=self.pdf_overlap
                    )
                    
                    for i, chunk in enumerate(chunks):
                        doc = {
                            "text": chunk,
                            "metadata": {
                                **pdf_metadata,
                                "chunk_id": i,
                                "total_chunks": len(chunks)
                            }
                        }
                        docs.append(doc)
                
                # Optionnel : Extraire texte des images avec OCR
                if OCR_SUPPORT and len(docs) < 2:  # Si peu de texte, essayer OCR
                    ocr_docs = self._extract_images_with_ocr(file_path, pdf_metadata)
                    docs.extend(ocr_docs)
        
        except Exception as e:
            print(f"   ⚠️  Erreur lecture PDF {file_path.name}: {e}")
        
        return docs
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Découpe un texte long en chunks avec chevauchement
        
        Args:
            text: Texte à découper
            chunk_size: Taille maximale d'un chunk
            overlap: Nombre de caractères en commun entre chunks
        
        Returns:
            Liste de chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Trouver la fin de phrase la plus proche
            if end < len(text):
                # Chercher un point suivi d'espace ou fin de ligne
                sentence_end = text.rfind('. ', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('.\n', start, end)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Avancer avec chevauchement
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def _extract_images_with_ocr(self, file_path: Path, metadata: Dict) -> List[Dict]:
        """
        Extrait le texte des images d'un PDF avec OCR (optionnel)
        
        Args:
            file_path: Chemin du PDF
            metadata: Métadonnées du document
        
        Returns:
            Liste de documents issus de l'OCR
        """
        if not OCR_SUPPORT:
            return []
        
        docs = []
        
        try:
            # Convertir PDF en images
            images = convert_from_path(str(file_path), dpi=200)
            
            for i, image in enumerate(images, 1):
                # Extraire texte avec Tesseract
                ocr_text = pytesseract.image_to_string(image, lang='fra+eng')
                
                if ocr_text.strip():
                    doc = {
                        "text": f"[Image/Tableau Page {i}]\n{ocr_text}",
                        "metadata": {
                            **metadata,
                            "source_type": "ocr",
                            "page": i
                        }
                    }
                    docs.append(doc)
        
        except Exception as e:
            print(f"   ⚠️  Erreur OCR: {e}")
        
        return docs
    
    def load_txt_file(self, file_path: Path, category: str = None) -> List[Dict]:
        """Charge un fichier TXT"""
        docs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) < 20:
                    continue
                
                doc = {
                    "text": paragraph,
                    "metadata": {
                        "source": file_path.name,
                        "type": "txt",
                        "category": category or file_path.parent.name,
                        "chunk_id": f"{file_path.stem}_chunk_{i}",
                        "file_path": str(file_path)
                    }
                }
                docs.append(doc)
        
        except Exception as e:
            print(f"   ⚠️  Erreur lecture TXT {file_path.name}: {e}")
        
        return docs
    
    def load_csv_file(self, file_path: Path, category: str = None) -> List[Dict]:
        """Charge un fichier CSV"""
        docs = []
        
        try:
            df = pd.read_csv(file_path)
            
            for idx, row in df.iterrows():
                text_parts = []
                metadata = {
                    "source": file_path.name,
                    "type": "csv",
                    "category": category or file_path.parent.name,
                    "row_id": idx
                }
                
                for col in df.columns:
                    if pd.notna(row[col]):
                        value = str(row[col])
                        text_parts.append(f"{col}: {value}")
                        metadata[col] = value
                
                doc = {
                    "text": " | ".join(text_parts),
                    "metadata": metadata
                }
                docs.append(doc)
        
        except Exception as e:
            print(f"   ⚠️  Erreur lecture CSV {file_path.name}: {e}")
        
        return docs
    
    def load_all_files(self):

        print(f"\n📂 Chargement des fichiers depuis {self.csv_folder}...")
        
        base_path = Path(self.csv_folder)
        
        if not base_path.exists():
            raise FileNotFoundError(f"❌ Dossier '{self.csv_folder}' introuvable")
        
        stats = {
            "csv": 0,
            "txt": 0,
            "pdf": 0,
            "total_docs": 0,
            "categories": set()
        }
        
        for file_path in base_path.rglob('*'):
            if file_path.is_file():
                category = file_path.parent.name if file_path.parent != base_path else "general"
                stats["categories"].add(category)
                
                suffix = file_path.suffix.lower()
                
                if suffix == '.csv':
                    docs = self.load_csv_file(file_path, category)
                    stats["csv"] += 1
                    print(f"   ✓ CSV: {category}/{file_path.name} ({len(docs)} entrées)")
                
                elif suffix in ['.txt', '.md']:
                    docs = self.load_txt_file(file_path, category)
                    stats["txt"] += 1
                    print(f"   ✓ TXT: {category}/{file_path.name} ({len(docs)} chunks)")
                
                elif suffix == '.pdf':
                    docs = self.load_pdf_file(file_path, category)
                    stats["pdf"] += 1
                    if docs:
                        print(f"   ✓ PDF: {category}/{file_path.name} ({len(docs)} chunks)")
                    else:
                        print(f"   ⚠️  PDF: {category}/{file_path.name} (vide ou erreur)")
                
                else:
                    continue
                
                self.documents.extend(docs)
                stats["total_docs"] += len(docs)
        
        print(f"\n📊 Statistiques de chargement:")
        print(f"   • Fichiers CSV: {stats['csv']}")
        print(f"   • Fichiers TXT: {stats['txt']}")
        print(f"   • Fichiers PDF: {stats['pdf']}")
        print(f"   • Catégories: {len(stats['categories'])} ({', '.join(sorted(stats['categories']))})")
        print(f"   • Documents totaux: {stats['total_docs']}")
        
        if stats["total_docs"] == 0:
            raise ValueError("❌ Aucun document chargé !")
    
    def create_vector_index(self, save_path: str = "vector_index"):
        """Crée l'index vectoriel FAISS"""
        print("\n🔧 Création de l'index vectoriel...")
        
        if not self.documents:
            raise ValueError("❌ Aucun document à indexer !")
        
        texts = [doc["text"] for doc in self.documents]
        print(f"   📊 Génération des embeddings pour {len(texts)} documents...")
        self.embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, f"{save_path}/faiss.index")
        
        with open(f"{save_path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"✅ Index vectoriel sauvegardé dans '{save_path}'")
    
    def load_vector_index(self, save_path: str = "vector_index"):
        """Charge un index vectoriel existant"""
        try:
            self.index = faiss.read_index(f"{save_path}/faiss.index")
            with open(f"{save_path}/documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            print(f"✅ Index vectoriel chargé depuis '{save_path}' ({len(self.documents)} docs)")
            return True
        except:
            print(f"⚠️  Aucun index trouvé dans '{save_path}'")
            return False
    
    def search(self, query: str, top_k: int = 5, category_filter: str = None) -> List[Dict]:
        """Recherche avec filtrage optionnel"""
        query_embedding = self.embedder.encode([query])
        
        search_k = top_k * 3 if category_filter else top_k
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(distance)
                
                if category_filter:
                    if doc["metadata"].get("category") == category_filter:
                        results.append(doc)
                else:
                    results.append(doc)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def generate_response(self, user_message: str, context_docs: List[Dict]) -> str:
        """Génère une réponse avec contexte"""
        
        categories = {}
        for doc in context_docs:
            cat = doc["metadata"].get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(doc)
        
        context_parts = []
        for cat, docs in categories.items():
            context_parts.append(f"[Catégorie: {cat}]")
            for i, doc in enumerate(docs, 1):
                source = doc["metadata"].get("source", "Inconnu")
                doc_type = doc["metadata"].get("type", "")
                
                # Indication spéciale pour les PDFs
                if doc_type == "pdf":
                    page_info = doc["metadata"].get("chunk_id", "")
                    context_parts.append(f"Document {i} ({source} - PDF, chunk {page_info}):\n{doc['text']}")
                else:
                    context_parts.append(f"Document {i} ({source}):\n{doc['text']}")
        
        context_text = "\n\n".join(context_parts)
        
        system_prompt = f"""Tu es un assistant conversationnel expert sur l'initiative citoyenne de Africtivistes en Afrique: CitizenLab.

CONTEXTE PERTINENT (organisé par catégorie):
{context_text}

INSTRUCTIONS:
- Réponds en français de manière naturelle et conversationnelle
- Base-toi PRIORITAIREMENT sur le contexte fourni ci-dessus 
- Pour les PDFs, le contexte peut provenir de différentes parties du document
- Si la question n'est pas dans le contexte, dis que tu n'as pas d'informations à ce sujet
- Sois précis avec les chiffres et les faits
- Utilise un ton amical et accessible"""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            
            assistant_message = response.choices[0].message.content
            
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        
        except Exception as e:
            return f"❌ Erreur: {str(e)}"
    
    def chat(self, user_message: str, top_k: int = 5, show_sources: bool = True, category_filter: str = None) -> str:
        """Interface principale du chatbot"""
        
        relevant_docs = self.search(user_message, top_k=top_k, category_filter=category_filter)
        response = self.generate_response(user_message, relevant_docs)
        
        # if show_sources and relevant_docs:
        #     sources = "\n\n📚 Sources utilisées:"
        #     seen_sources = set()
        #     for i, doc in enumerate(relevant_docs[:3], 1):
        #         source = doc['metadata'].get('source', 'Inconnu')
        #         category = doc['metadata'].get('category', 'general')
        #         doc_type = doc['metadata'].get('type', '')
                
        #         source_key = f"{category}/{source}"
        #         if source_key not in seen_sources:
        #             type_icon = "📄" if doc_type == "pdf" else "📝" if doc_type == "txt" else "📊"
        #             sources += f"\n   {i}. {type_icon} [{category}] {source}"
        #             seen_sources.add(source_key)
        #     response += sources
        
        return response
    
    def reset_conversation(self):
        """Réinitialise l'historique"""
        self.conversation_history = []
        print("🔄 Conversation réinitialisée")


def main():
    """Fonction de test"""
    print("=" * 60)
    print("🤖 CHATBOT RAG CITIZENLAB - VERSION PDF")
    print("=" * 60)
    
    chatbot = ChatbotRAG(csv_folder="knowledge_base")
    
    if not chatbot.load_vector_index():
        chatbot.load_all_files()
        chatbot.create_vector_index()
    
    print("\n" + "=" * 60)
    print("💬 Conversation démarrée!")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("Vous: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Au revoir!")
            break
        
        if user_input.lower() == 'reset':
            chatbot.reset_conversation()
            continue
        
        print("\n🤖 Assistant: ", end="")
        response = chatbot.chat(user_input, show_sources=True)
        print(response)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
