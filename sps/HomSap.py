"""
Catalog definitions for Homo Sapiens
Modified from https://github.com/popsim-consortium/stdpopsim/tree/main/stdpopsim/catalog
"""
import math
import logging

import msprime

import sps.genomes as genomes
import sps.species as species
import sps.genetic_maps as genetic_maps
import sps.models as models

logger = logging.getLogger(__name__)

###########################################################
#
# Genome definition
#
###########################################################

# List of chromosomes.

# FIXME: add mean mutation rate data to this table.
# Name  Length  mean_recombination_rate mean_mutation_rate

# length information can be found here
# <http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/chromInfo.txt.gz>

# mean_recombination_rate was computed across all windows of the GRCh37 genetic map
# <ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/working/20110106_recombination_hotspots>
_chromosome_data = """\
chr1 	 249250621 	 1.1485597641285933e-08
chr2 	 243199373 	 1.1054289277533446e-08
chr3 	 198022430 	 1.1279585624662551e-08
chr4 	 191154276 	 1.1231162636001008e-08
chr5 	 180915260 	 1.1280936570022824e-08
chr6 	 171115067 	 1.1222852661225285e-08
chr7 	 159138663 	 1.1764614397655721e-08
chr8 	 146364022 	 1.1478465778920576e-08
chr9 	 141213431 	 1.1780701596308656e-08
chr10 	 135534747 	 1.3365134257075317e-08
chr11 	 135006516 	 1.1719334320833283e-08
chr12 	 133851895 	 1.305017186986983e-08
chr13 	 115169878 	 1.0914860554958317e-08
chr14 	 107349540 	 1.119730771394731e-08
chr15 	 102531392 	 1.3835785893339787e-08
chr16 	 90354753 	 1.4834607113882717e-08
chr17 	 81195210 	 1.582489036239487e-08
chr18 	 78077248 	 1.5075956950023575e-08
chr19 	 59128983 	 1.8220141872466202e-08
chr20 	 63025520 	 1.7178269031631664e-08
chr21 	 48129895 	 1.3045214034879191e-08
chr22 	 51304566 	 1.4445022767788226e-08
chrX 	 155270560 	 1.164662223273842e-08
chrY 	 59373566 	 0.0
"""


'''_genome2001 = stdpopsim.Citation(
    doi="http://dx.doi.org/10.1038/35057062",
    year="2001",
    author="The Genome Sequencing Consortium",
    reasons={stdpopsim.CiteReason.ASSEMBLY}
)

_hapmap2007 = stdpopsim.Citation(
    doi="https://doi.org/10.1038/nature06258",
    year=2007,
    author="The International HapMap Consortium",
)

_tian2019 = stdpopsim.Citation(
    doi="https://doi.org/10.1016/j.ajhg.2019.09.012",
    year="2019",
    author="Tian, Browning, and Browning",
    reasons={stdpopsim.CiteReason.MUT_RATE}
)

_tremblay2000 = stdpopsim.Citation(
    doi="https://doi.org/10.1086/302770",
    year="2000",
    author="Tremblay and Vezina",
    reasons={stdpopsim.CiteReason.GEN_TIME}
)

_takahata1993 = stdpopsim.Citation(
    doi="https://doi.org/10.1093/oxfordjournals.molbev.a039995",
    year="1993",
    author="Takahata",
    reasons={stdpopsim.CiteReason.POP_SIZE}
)'''

_chromosomes = []
for line in _chromosome_data.splitlines():
    name, length, mean_rr = line.split()[:3]
    _chromosomes.append(genomes.Chromosome(
        id=name, length=int(length),
        mutation_rate=1.29e-8, #2.35e-8,
        recombination_rate=float(mean_rr)))

_genome = genomes.Genome(chromosomes=_chromosomes)
'''mutation_rate_citations=[
    _tian2019.because(stdpopsim.CiteReason.MUT_RATE)],
recombination_rate_citations=[
    _hapmap2007.because(stdpopsim.CiteReason.REC_RATE)],
assembly_citations=[
    _genome2001]'''

_species = species.Species(
    id="HomSap",
    name="Homo sapiens",
    common_name="Human",
    genome=_genome,
    generation_time=30)
'''generation_time_citations=[
    _tremblay2000.because(stdpopsim.CiteReason.GEN_TIME)],
population_size=10**4,
population_size_citations=[
    _takahata1993.because(stdpopsim.CiteReason.POP_SIZE)]'''


species.register_species(_species)


###########################################################
#
# Genetic maps
#
###########################################################


_gm = genetic_maps.GeneticMap(
    species=_species,
    id="HapMapII_GRCh37",
    description="HapMap Phase II lifted over to GRCh37",
    long_description="""
        This genetic map is from the Phase II Hapmap project
        and based on 3.1 million genotyped SNPs
        from 270 individuals across four populations (YRI, CEU, CHB and JPT).
        Genome wide recombination rates were estimated using LDHat.
        This version of the HapMap genetic map was lifted over to GRCh37
        (and adjusted in regions where the genome assembly had rearranged)
        for use in the 1000 Genomes project. Please see the README file on
        the 1000 Genomes download site for details of these adjustments.
        """,
    url=(
        "https://stdpopsim.s3-us-west-2.amazonaws.com/genetic_maps/"
        "HomSap/HapmapII_GRCh37_RecombinationHotspots.tar.gz"),
    file_pattern="genetic_map_GRCh37_{id}.txt"
    '''citations=[
        _hapmap2007.because(stdpopsim.CiteReason.GEN_MAP)],'''
    )
_species.add_genetic_map(_gm)

_gm = genetic_maps.GeneticMap(
    species=_species,
    id="DeCodeSexAveraged_GRCh36",
    description="Sex averaged map from deCode family study",
    long_description="""
        This genetic map is from the deCode study of recombination
        events in 15,257 parent-offspring pairs from Iceland.
        289,658 phased autosomal SNPs were used to call recombinations
        within these families, and recombination rates computed from the
        density of these events. This is the combined male and female
        (sex averaged) map. See
        https://www.decode.com/addendum/ for more details.""",
    url=(
        "https://stdpopsim.s3-us-west-2.amazonaws.com/genetic_maps/"
        "HomSap/decode_2010_sex-averaged_map.tar.gz"),
    file_pattern="genetic_map_decode_2010_sex-averaged_{id}.txt"
    '''citations=[
        stdpopsim.Citation(
            year=2010,
            author="Kong et al",
            doi="https://doi.org/10.1038/nature09525",
            reasons={stdpopsim.CiteReason.GEN_MAP})]'''
    )
_species.add_genetic_map(_gm)


###########################################################
#
# Demographic models
#
###########################################################

# population definitions that are reused.
_yri_population = models.Population(
    id="YRI",
    description="1000 Genomes YRI (Yorubans)")
_ceu_population = models.Population(
    id="CEU",
    description=(
        "1000 Genomes CEU (Utah Residents (CEPH) with Northern and "
        "Western European Ancestry"))
_chb_population = models.Population(
    id="CHB",
    description="1000 Genomes CHB (Han Chinese in Beijing, China)")


'''_tennessen_et_al = stdpopsim.Citation(
    author="Tennessen et al.",
    year=2012,
    doi="https://doi.org/10.1126/science.1219240",
    reasons={stdpopsim.CiteReason.DEM_MODEL})'''


# adding flexible parameters to infer
def ooa_3(N_A=7300, N_B=2100, N_AF=12300, N_EU0=1000, N_AS0=510, r_EU=0.004, \
    r_AS=0.0055, T_AF=8800, T_B=5600, T_EU_AS=848, m_AF_B=25e-5, m_AF_EU=3e-5,
    m_AF_AS=1.9e-5, m_EU_AS=9.6e-5):
    id = "OutOfAfrica_3G09"
    description = "Three population out-of-Africa"
    long_description = """
        The three population Out-of-Africa model from Gutenkunst et al. 2009.
        It describes the ancestral human population in Africa, the out of Africa
        event, and the subsequent European-Asian population split.
        Model parameters are the maximum likelihood values of the
        various parameters given in Table 1 of Gutenkunst et al.
    """
    populations = [
        _yri_population,
        _ceu_population,
        _chb_population
    ]

    '''citations = [stdpopsim.Citation(
        author="Gutenkunst et al.",
        year=2009,
        doi="https://doi.org/10.1371/journal.pgen.1000695",
        reasons={stdpopsim.CiteReason.DEM_MODEL})
    ]'''

    generation_time = 25

    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1.
    #N_A = 7300
    #N_B = 2100
    #N_AF = 12300
    #N_EU0 = 1000
    #N_AS0 = 510
    # Times are provided in years, so we convert into generations.

    #T_AF = 8800
    #T_B = 5600
    #T_EU_AS = 848
    # We need to work out the starting (diploid) population sizes based on
    # the growth rates provided for these two populations
    #r_EU = 0.004
    #r_AS = 0.0055
    N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
    N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
    # Migration rates during the various epochs.
    #m_AF_B = 25e-5
    #m_AF_EU = 3e-5
    #m_AF_AS = 1.9e-5
    #m_EU_AS = 9.6e-5

    return models.DemographicModel(
        id=id,
        description=description,
        long_description=long_description,
        populations=populations,
        #citations=citations,
        generation_time=generation_time,

        # Population IDs correspond to their indexes in the population
        # configuration array. Therefore, we have 0=YRI, 1=CEU and 2=CHB
        # initially.
        population_configurations=[
            msprime.PopulationConfiguration(
                initial_size=N_AF, metadata=populations[0].asdict()),
            msprime.PopulationConfiguration(
                initial_size=N_EU, growth_rate=r_EU,
                metadata=populations[1].asdict()),
            msprime.PopulationConfiguration(
                initial_size=N_AS, growth_rate=r_AS,
                metadata=populations[2].asdict()),
        ],
        migration_matrix=[
            [      0, m_AF_EU, m_AF_AS],  # noqa
            [m_AF_EU,       0, m_EU_AS],  # noqa
            [m_AF_AS, m_EU_AS,       0],  # noqa
        ],
        demographic_events=[
            # CEU and CHB merge into B with rate changes at T_EU_AS
            msprime.MassMigration(
                time=T_EU_AS, source=2, destination=1, proportion=1.0),
            msprime.MigrationRateChange(time=T_EU_AS, rate=0),
            msprime.MigrationRateChange(
                time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
            msprime.MigrationRateChange(
                time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
            msprime.PopulationParametersChange(
                time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
            # Population B merges into YRI at T_B
            msprime.MassMigration(
                time=T_B, source=1, destination=0, proportion=1.0),
            msprime.MigrationRateChange(
                time=T_B, rate=0),
            # Size changes to N_A at T_AF
            msprime.PopulationParametersChange(
                time=T_AF, initial_size=N_A, population_id=0)
        ],
        )


# don't add right away, want to change as we go
#_species.add_demographic_model(_ooa_3())
