from utilities.parser import arff
import os

datasets = {
    'enron': (
        False,
        '/enron/Enron_corr_trainvalid.arff',
        '/enron/Enron_corr_trainvalid.arff',
        '/enron/Enron_corr_test.arff'
    ),
    'imclef07a': (
        False,
        '/ImCLEF07A_Train.arff',
        '/ImCLEF07A_Train.arff',
        '/ImCLEF07A_Test.arff'
    ),
    'cellcycle_FUN': (
        False,
        '/datasets_FUN/cellcycle_FUN/cellcycle_FUN.train.arff',
        '/datasets_FUN/cellcycle_FUN/cellcycle_FUN.valid.arff',
        '/datasets_FUN/cellcycle_FUN/cellcycle_FUN.test.arff'
    ),
    'church_FUN': (
        False,
        '/datasets_FUN/church_FUN/church_FUN.train.arff',
        '/datasets_FUN/church_FUN/church_FUN.valid.arff',
        '/datasets_FUN/church_FUN/church_FUN.test.arff'
    ),
    'derisi_FUN': (
        False,
        '/datasets_FUN/derisi_FUN/derisi_FUN.train.arff',
        '/datasets_FUN/derisi_FUN/derisi_FUN.valid.arff',
        '/datasets_FUN/derisi_FUN/derisi_FUN.test.arff'
    ),
    'eisen_FUN': (
        False,
        '/datasets_FUN/eisen_FUN/eisen_FUN.train.arff',
        '/datasets_FUN/eisen_FUN/eisen_FUN.valid.arff',
        '/datasets_FUN/eisen_FUN/eisen_FUN.test.arff'
    ),
    'expr_FUN': (
        False,
        '/datasets_FUN/expr_FUN/expr_FUN.train.arff',
        '/datasets_FUN/expr_FUN/expr_FUN.valid.arff',
        '/datasets_FUN/expr_FUN/expr_FUN.test.arff'
    ),
    'gasch1_FUN': (
        False,
        '/datasets_FUN/gasch1_FUN/gasch1_FUN.train.arff',
        '/datasets_FUN/gasch1_FUN/gasch1_FUN.valid.arff',
        '/datasets_FUN/gasch1_FUN/gasch1_FUN.test.arff'
    ),
    'gasch2_FUN': (
        False,
        '/datasets_FUN/gasch2_FUN/gasch2_FUN.train.arff',
        '/datasets_FUN/gasch2_FUN/gasch2_FUN.valid.arff',
        '/datasets_FUN/gasch2_FUN/gasch2_FUN.test.arff'
    ),
    'hom_FUN': (
        False,
        '/datasets_FUN/hom_FUN/hom_FUN.train.arff',
        '/datasets_FUN/hom_FUN/hom_FUN.valid.arff',
        '/datasets_FUN/hom_FUN/hom_FUN.test.arff'
    ),
    'pheno_FUN': (
        False,
        '/datasets_FUN/pheno_FUN/pheno_FUN.train.arff',
        '/datasets_FUN/pheno_FUN/pheno_FUN.valid.arff',
        '/datasets_FUN/pheno_FUN/pheno_FUN.test.arff'
    ),
    'seq_FUN': (
        False,
        '/datasets_FUN/seq_FUN/seq_FUN.train.arff',
        '/datasets_FUN/seq_FUN/seq_FUN.valid.arff',
        '/datasets_FUN/seq_FUN/seq_FUN.test.arff'
    ),
    'spo_FUN': (
        False,
        '/datasets_FUN/spo_FUN/spo_FUN.train.arff',
        '/datasets_FUN/spo_FUN/spo_FUN.valid.arff',
        '/datasets_FUN/spo_FUN/spo_FUN.test.arff'
    ),
    'struc_FUN': (
        False,
        '/datasets_FUN/struc_FUN/struc_FUN.train.arff',
        '/datasets_FUN/struc_FUN/struc_FUN.valid.arff',
        '/datasets_FUN/struc_FUN/struc_FUN.test.arff'
    ),
    'cellcycle_GO': (
        True,
        '/datasets_GO/cellcycle_GO/cellcycle_GO.train.arff',
        '/datasets_GO/cellcycle_GO/cellcycle_GO.valid.arff',
        '/datasets_GO/cellcycle_GO/cellcycle_GO.test.arff'
    ),
    'church_GO': (
        True,
        '/datasets_GO/church_GO/church_GO.train.arff',
        '/datasets_GO/church_GO/church_GO.valid.arff',
        '/datasets_GO/church_GO/church_GO.test.arff'
    ),
    'derisi_GO': (
        True,
        '/datasets_GO/derisi_GO/derisi_GO.train.arff',
        '/datasets_GO/derisi_GO/derisi_GO.valid.arff',
        '/datasets_GO/derisi_GO/derisi_GO.test.arff'
    ),
    'eisen_GO': (
        True,
        '/datasets_GO/eisen_GO/eisen_GO.train.arff',
        '/datasets_GO/eisen_GO/eisen_GO.valid.arff',
        '/datasets_GO/eisen_GO/eisen_GO.test.arff'
    ),
    'expr_GO': (
        True,
        '/datasets_GO/expr_GO/expr_GO.train.arff',
        '/datasets_GO/expr_GO/expr_GO.valid.arff',
        '/datasets_GO/expr_GO/expr_GO.test.arff'
    ),
    'gasch1_GO': (
        True,
        '/datasets_GO/gasch1_GO/gasch1_GO.train.arff',
        '/datasets_GO/gasch1_GO/gasch1_GO.valid.arff',
        '/datasets_GO/gasch1_GO/gasch1_GO.test.arff'
    ),
    'gasch2_GO': (
        True,
        '/datasets_GO/gasch2_GO/gasch2_GO.train.arff',
        '/datasets_GO/gasch2_GO/gasch2_GO.valid.arff',
        '/datasets_GO/gasch2_GO/gasch2_GO.test.arff'
    ),
    'hom_GO': (
        True,
        '/datasets_GO/hom_GO/hom_GO.train.arff',
        '/datasets_GO/hom_GO/hom_GO.valid.arff',
        '/datasets_GO/hom_GO/hom_GO.test.arff'
    ),
    'pheno_GO': (
        True,
        '/datasets_GO/pheno_GO/pheno_GO.train.arff',
        '/datasets_GO/pheno_GO/pheno_GO.valid.arff',
        '/datasets_GO/pheno_GO/pheno_GO.test.arff'
    ),
    'seq_GO': (
        True,
        '/datasets_GO/seq_GO/seq_GO.train.arff',
        '/datasets_GO/seq_GO/seq_GO.valid.arff',
        '/datasets_GO/seq_GO/seq_GO.test.arff'
    ),
    'spo_GO': (
        True,
        '/datasets_GO/spo_GO/spo_GO.train.arff',
        '/datasets_GO/spo_GO/spo_GO.valid.arff',
        '/datasets_GO/spo_GO/spo_GO.test.arff'
    ),
    'struc_GO': (
        True,
        '/datasets_GO/struc_GO/struc_GO.train.arff',
        '/datasets_GO/struc_GO/struc_GO.valid.arff',
        '/datasets_GO/struc_GO/struc_GO.test.arff'
    ),
}


def initialize_dataset(data_folder, name):
    if name in datasets:
        d = datasets[name]
        return (
            arff(os.path.join(data_folder, d[1]), d[0], True),
            arff(os.path.join(data_folder, d[2]), d[0]),
            arff(os.path.join(data_folder, d[3]), d[0]),
        )
