"""Process-only mapped compressed embedding; original rows and dequant arithmetic intact."""
import torch
from freetoken.kernel.pinned import alloc_pinned_tensor
from mapped_host_bytes import MappedHostBytes


def enable():
    """Move the table during eager warmup, before any captured graph can retain its old pointer."""
    from freetoken.models.qwen4_exp.gguf_weight import GGUFEmbedding
    original=GGUFEmbedding.forward
    records=[]

    def forward(self,ids):
        """Own pinned bytes and their CUDA alias until this embedding and its graph readers drain."""
        if not hasattr(self,'_mapped_host_interface'):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Mapped embedding must bind before CUDA graph capture')
            from freetoken.kernel.gguf_quant import dequantize_rows
            packed=self.packed_0
            before=torch.cuda.mem_get_info()[0]
            allocated_before=torch.cuda.memory_allocated()
            host=alloc_pinned_tensor(packed.numel(),dtype=torch.uint8)
            host.copy_(packed.cpu())
            interface=MappedHostBytes(host)
            alias=interface.alias(packed.device)
            check_ids=torch.tensor([0,self.num_embeddings-1,self.num_embeddings//2],device=packed.device,dtype=torch.int32)
            reference=dequantize_rows(packed,check_ids,self.embedding_dim,self._type)
            mapped=dequantize_rows(alias,check_ids,self.embedding_dim,self._type)
            if not torch.equal(reference,mapped):
                raise RuntimeError('Mapped original embedding rows changed values')
            self._mapped_host_interface=interface
            self.packed_0=alias
            owned_bytes=packed.numel()
            del packed,reference,mapped,check_ids
            torch.cuda.empty_cache()
            records.append(dict(bytes=owned_bytes,vocab=self.num_embeddings,width=self.embedding_dim,
                type_id=self._type,actual_boundary_rows_exact=True,
                cuda_alias_ptr=alias.data_ptr(),device_visible_ptr=interface.ptr,host_ptr=host.data_ptr(),
                free_before=before,free_after=torch.cuda.mem_get_info()[0],
                allocated_before=allocated_before,allocated_after=torch.cuda.memory_allocated()))
        return original(self,ids)

    GGUFEmbedding.forward=forward
    return records

