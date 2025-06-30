#!/usr/bin/env python3
"""
FrameNet Scenario Frame Analysis Scripts
========================================

This script generates CSV files for scenario frame analysis from FrameNet Brasil database.
It creates two main outputs:
1. scenario_frame_groupings.csv - Frame-to-frame relationships
2. scenario_frame_fe_relationships.csv - Frame Element relationships

Requirements:
- pandas
- All CSV files in the same directory as this script

Usage:
    python framenet_analysis.py

The script will read the input CSV files and generate the analysis outputs.
"""

import pandas as pd
import sys
from pathlib import Path


def load_data():
    """Load all FrameNet CSV files and return as DataFrames."""
    try:
        print("Loading FrameNet data files...")

        # Read all CSV files
        frame_relations = pd.read_csv('frame_relations.csv')
        fe_relations = pd.read_csv('fe_relations.csv')
        fes = pd.read_csv('fes.csv')
        scenario_frames = pd.read_csv('scenarios_frames.csv')
        all_frames = pd.read_csv('all_frames.csv')

        print(f"✓ Loaded {len(frame_relations)} frame relations")
        print(f"✓ Loaded {len(fe_relations)} FE relations")
        print(f"✓ Loaded {len(fes)} frame elements")
        print(f"✓ Loaded {len(scenario_frames)} scenario frames")
        print(f"✓ Loaded {len(all_frames)} total frames")

        return frame_relations, fe_relations, fes, scenario_frames, all_frames

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file {e.filename}")
        print("Please ensure all CSV files are in the same directory as this script:")
        print("- frame_relations.csv")
        print("- fe_relations.csv")
        print("- fes.csv")
        print("- scenarios_frames.csv")
        print("- all_frames.csv")
        sys.exit(1)


def create_frame_groupings(frame_relations, scenario_frames, all_frames):
    """Create scenario frame groupings CSV."""
    print("\n📊 Generating scenario frame groupings...")

    # Create frame name lookup
    frame_names = dict(zip(all_frames['idFrame'], all_frames['name']))
    scenario_names = dict(zip(scenario_frames['idFrame'], scenario_frames['name']))

    # Get scenario frame IDs
    scenario_frame_ids = set(scenario_frames['idFrame'])

    # Filter relations where scenario frames are parents
    scenario_parent_relations = frame_relations[
        frame_relations['parent'].isin(scenario_frame_ids)
    ].copy()

    # Add names
    scenario_parent_relations['scenarioName'] = scenario_parent_relations['parent'].map(
        lambda x: scenario_names.get(x, frame_names.get(x, 'Unknown'))
    )
    scenario_parent_relations['childFrameName'] = scenario_parent_relations['child'].map(
        lambda x: frame_names.get(x, 'Unknown')
    )

    # Create final dataframe
    result = scenario_parent_relations[['parent', 'child', 'scenarioName', 'childFrameName']].copy()
    result.columns = ['idFrameScenario', 'idFrameChild', 'scenarioName', 'childFrameName']

    print(f"✓ Generated {len(result)} scenario frame relationships")
    print(f"✓ {len(result['idFrameScenario'].unique())} scenario frames have children")

    return result


def create_fe_relationships(frame_relations, fe_relations, fes, scenario_frames, all_frames):
    """Create detailed Frame Element relationships CSV."""
    print("\n🔗 Generating Frame Element relationships...")

    # Create lookups
    frame_names = dict(zip(all_frames['idFrame'], all_frames['name']))
    scenario_names = dict(zip(scenario_frames['idFrame'], scenario_frames['name']))

    # Create FE lookup
    fe_lookup = {}
    for _, fe in fes.iterrows():
        fe_lookup[fe['idFrameElement']] = {
            'idFrame': fe['idFrame'],
            'frameName': fe['frameName'],
            'feName': fe['FrameElementName'],
            'coreType': fe['coreType']
        }

    scenario_frame_ids = set(scenario_frames['idFrame'])

    # Get scenario parent relations
    scenario_parent_relations = frame_relations[
        frame_relations['parent'].isin(scenario_frame_ids)
    ]

    results = []

    print("Processing frame relationships...")
    for idx, frame_rel in scenario_parent_relations.iterrows():
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(scenario_parent_relations)} frame relations...")

        scenario_name = scenario_names.get(frame_rel['parent'],
                                           frame_names.get(frame_rel['parent'], 'Unknown'))
        child_name = frame_names.get(frame_rel['child'], 'Unknown')

        # Find FE relations between these frames
        relevant_fe_relations = fe_relations[
            (fe_relations['parent'].isin([fe_id for fe_id, fe_info in fe_lookup.items()
                                          if fe_info['idFrame'] == frame_rel['parent']])) &
            (fe_relations['child'].isin([fe_id for fe_id, fe_info in fe_lookup.items()
                                         if fe_info['idFrame'] == frame_rel['child']]))
            ]

        if len(relevant_fe_relations) > 0:
            for _, fe_rel in relevant_fe_relations.iterrows():
                parent_fe = fe_lookup.get(fe_rel['parent'])
                child_fe = fe_lookup.get(fe_rel['child'])

                if parent_fe and child_fe:
                    results.append({
                        'idFrameScenario': frame_rel['parent'],
                        'idFrameChild': frame_rel['child'],
                        'scenarioName': scenario_name,
                        'childFrameName': child_name,
                        'idFeScenario': fe_rel['parent'],
                        'idFeChild': fe_rel['child'],
                        'feScenarioName': parent_fe['feName'],
                        'feChildName': child_fe['feName'],
                        'feRelationType': fe_rel['relation']
                    })
        else:
            # Include frame relation without FE mappings
            results.append({
                'idFrameScenario': frame_rel['parent'],
                'idFrameChild': frame_rel['child'],
                'scenarioName': scenario_name,
                'childFrameName': child_name,
                'idFeScenario': '',
                'idFeChild': '',
                'feScenarioName': '',
                'feChildName': '',
                'feRelationType': ''
            })

    result_df = pd.DataFrame(results)

    print(f"✓ Generated {len(result_df)} total relationships")

    # Statistics
    with_fe = result_df[result_df['idFeScenario'] != '']
    without_fe = result_df[result_df['idFeScenario'] == '']

    print(f"✓ {len(with_fe)} relationships with FE mappings")
    print(f"✓ {len(without_fe)} relationships without FE mappings")

    if len(with_fe) > 0:
        fe_relation_types = with_fe['feRelationType'].value_counts()
        print("✓ FE relation types:")
        for rel_type, count in fe_relation_types.head().items():
            print(f"    {rel_type}: {count}")

    return result_df


def main():
    """Main execution function."""
    print("🚀 FrameNet Scenario Frame Analysis")
    print("=" * 40)

    # Load data
    frame_relations, fe_relations, fes, scenario_frames, all_frames = load_data()

    # Generate scenario frame groupings
    frame_groupings = create_frame_groupings(frame_relations, scenario_frames, all_frames)

    # Save frame groupings
    output_file1 = 'scenario_frame_groupings.csv'
    frame_groupings.to_csv(output_file1, index=False)
    print(f"💾 Saved frame groupings to: {output_file1}")

    # Generate FE relationships
    fe_relationships = create_fe_relationships(frame_relations, fe_relations, fes,
                                               scenario_frames, all_frames)

    # Save FE relationships
    output_file2 = 'scenario_frame_fe_relationships.csv'
    fe_relationships.to_csv(output_file2, index=False)
    print(f"💾 Saved FE relationships to: {output_file2}")

    print("\n✅ Analysis complete!")
    print(f"📁 Generated files:")
    print(f"   - {output_file1} ({len(frame_groupings)} rows)")
    print(f"   - {output_file2} ({len(fe_relationships)} rows)")

    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   - Total scenario frames: {len(scenario_frames)}")
    print(f"   - Scenario frames with children: {len(frame_groupings['idFrameScenario'].unique())}")
    print(
        f"   - Average children per scenario: {len(frame_groupings) / len(frame_groupings['idFrameScenario'].unique()):.1f}")

    with_fe = fe_relationships[fe_relationships['idFeScenario'] != '']
    if len(with_fe) > 0:
        print(f"   - FE relationship coverage: {len(with_fe) / len(fe_relationships) * 100:.1f}%")


if __name__ == "__main__":
    main()