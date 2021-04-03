import { gql } from '@apollo/client';

export const ENCOUNTER = gql`
    query ($id: token) {
        Encounter(_id: $id) {
            id
            status
            meta {
            lastUpdated
            }
        }
    }
 `;

export const ENCOUNTERS_BY_PATIENT_ID = gql`
    query ($patientId: token) {
        EncounterByPatientId(_id: $patientId) {
            entries
        }
    }
 `;
