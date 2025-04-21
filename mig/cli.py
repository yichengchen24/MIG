import typer

from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

from .data import load_dataset, dump_dataset
from .label import LabelGraphType, create_label_graph
from .sampler import SamplerType, create_sampler


cli = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@cli.command("hello")
def hello():
    typer.echo("Hello World")
    

@cli.command("sample")
def sample(
    src: Annotated[Path, typer.Argument(help="Path to the data pool")],
    out: Annotated[Optional[Path], typer.Option(help="Path to save sampled data")] = None,
    num_sample: Annotated[int, typer.Option("-n", "--num", help="Number of samples")] = 100,
    valid_tag_path: Annotated[str, typer.Option(help="Path to valid tags")] = "",
    label_graph_type: Annotated[LabelGraphType, typer.Option(help="Type of label graph to use")] = LabelGraphType.SIM,
    load_from: Annotated[Optional[str], typer.Option(help="")] = None,
    embedding_model: Annotated[str, typer.Option(help="Embedding model to use")] = "",
    sim_threshold: Annotated[float, typer.Option(help="Sim threshold to use")] = 0.9,
    sampler_type: Annotated[SamplerType, typer.Option(help="Type of sampler to use")] = SamplerType.MIG,
    phi_type: Annotated[str, typer.Option(help="Type of phi to use")] = "pow",
    phi_alpha: Annotated[float, typer.Option(help="Const to use in pow phi")] = 1.0,
    phi_a: Annotated[float, typer.Option(help="Const to use in pow phi")] = 1e-6,
    phi_b: Annotated[float, typer.Option(help="Exponent to use in pow phi")] = 0.8,
    prop_weight: Annotated[float, typer.Option(help="Propagation weight used in propagation matrix")] = 1.0,
    norm: Annotated[bool, typer.Option("--norm", help="Use normalize in sampler", is_flag=True)] = False,
    batch_size: Annotated[int, typer.Option(help="batch size used in sample")] = 0,
):
    """Sample data from pool"""
    
    typer.echo(f"Loadingg data from {src}")
    pool = load_dataset(src, valid_tag_path=valid_tag_path)
    
    typer.echo(f"Creating label graph using {label_graph_type} label graph")
    label_graph = create_label_graph(label_graph_type, dataset=pool, load_from=load_from, embedding_model=embedding_model, sim_threshold=sim_threshold)
    
    typer.echo(f"Creating sampler using {sampler_type} sampler")
    sampler = create_sampler(sampler_type, label_graph, phi_type, phi_alpha, phi_a, phi_b, prop_weight, norm)
    
    sampled = sampler.sample(pool, num_sample, batch_size=batch_size)
    
    if out is None:
        out = src.with_stem(f"{src.stem}-sampled-{num_sample}.jsonl")
    dump_dataset(sampled, out)


if __name__ == "__main__":
    cli()
