"""
Module for data pre-processing.
"""

from loguru import logger
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from model.pipeline.collection import load_data_from_csv


def get_descriptor_df(failed_output: bool = False) -> pd.DataFrame:
    """
    Main function to process data.

    This functions creates a pandas.DataFrame of molecular descriptors from
    RDKit.Chem.Descriptors

    Returns:
        pandas.DataFrame: DataFrame with 210 descriptors, with SMILS as indices
        and a label column with target.
    """
    df = load_data_from_csv()
    temp_dict = {}
    failed_smiles = []
    logger.info('Creating descriptors...')
    for smile in df['smiles']:
        temp_mol = Chem.MolFromSmiles(smile)
        if temp_mol:
            temp_dict[smile] = Descriptors.CalcMolDescriptors(temp_mol)
        else:
            failed_smiles.append(smile)
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    temp_df['label'] = df.set_index('smiles')['expt']

    if len(failed_smiles) > 0:
        logger.debug(f'{len(failed_smiles)} SMILES failed to be read as MOL.')
        if failed_output:
            print('Warning!\n' +
                  ' {} SMILES failed to convert.'.format(len(failed_smiles)))
            return temp_df, failed_smiles
        else:
            print('Warning!\n' +
                  '{} SMILES failed to convert.'.format(len(failed_smiles)))
            print('Run with failed_output=True for a list of failed SMILES.')
            return temp_df
    else:
        logger.info('No SMILES failed to convert to MOL.')
        return temp_df


if __name__ == '__main__':
    descriptors_df = get_descriptor_df()
    print(descriptors_df.head())
