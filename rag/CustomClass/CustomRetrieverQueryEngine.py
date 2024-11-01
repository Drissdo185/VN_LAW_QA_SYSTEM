from copy import deepcopy

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle, QueryType
from llama_index.core.base.response.schema import RESPONSE_TYPE, StreamingResponse, Response
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from pyvi.ViTokenizer import ViTokenizer



class CustomRetrieverQueryEngine(RetrieverQueryEngine):

    def query(self, str_or_query_bundle: QueryType, container_status=None)-> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"): #debug
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)

            return  self._query(str_or_query_bundle, container_status = container_status)

    def _query(self, query_bundle: QueryBundle,
               container_status=None) -> RESPONSE_TYPE:
        """"Get a query."""
        with self.callback_manager.event(CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            if container_status:
                container_status.update(label=f"**Context Retrieval: Retrieving Relevant Passages ...**")

            retrieval_query_bundle = deepcopy(query_bundle)
            retrieval_query_bundle.query_str = ViTokenizer.tokenize(retrieval_query_bundle.query_str.lower())
            nodes = self.retrieve(retrieval_query_bundle)

            if container_status:
                if not len(nodes):
                    container_status.markdown("----")
                    container_status.markdown(f"*There are no relevant passages*")

                for idx, node in enumerate(nodes):
                    metadata = node.metadata
                    container_status.markdown("----")
                    container_status.markdown(f"**Top {idx+1} Relevant Passage**")
                    parent_text = node.get_content()
                    start_char_idx = node.node.start_char_idx
                    end_char_idx = node.node.end_char_idx
                    container_status.markdown(parent_text[:start_char_idx] + ":green[" + parent_text[start_char_idx:end_char_idx].replace("\n", " ") + "]"+ parent_text[end_char_idx:],unsafe_allow_html=True)
                    container_status.markdown(f'**Retrieval scores**: *Dense score*: {metadata["dense_score"]:.3f}({int(metadata["dense_rank"])}).\
                                              *BM25 score*: {metadata["bm25_score"]:.3f}({int(metadata["bm25_rank"])}).\
                                              *Hybrid score*: {metadata["retrieval_score"]:.3f}({int(metadata["retrieval_rank"])}).\
                                              *Cross Encoder score*:  {node.score:.3f}({idx+1})')
                    container_status.markdown(f"**Source**: *{metadata['source']}*")
                    container_status.markdown(f"**Document**: *{metadata['filename']}*")

                container_status.update(label=f"**Context Retrieval**")
                container_status.update(state="complete")

            response = self._response_synthesizer.synthesize(
                # response synthesizer: response from LLM when it has data from retrieval
                query=query_bundle,
                nodes=nodes,
            )

            if (isinstance(response, Response) and response.response == "Empty Response"):
                response = StreamingResponse("Xin lỗi, tôi là trợ lý về pháp luật. Tôi không tìm thấy dữ liệu liên quan tới câu hỏi của bạn. Vui lòng hỏi các vấn đề liên quan tới pháp luật để tôi có thể trợ giúp bạn.")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response




