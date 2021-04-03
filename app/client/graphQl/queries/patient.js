import { gql } from '@apollo/client';

export const ALL_PATIENTS_QUERY = gql`
    {
        AllPatientsQuery {
            entries
        }
    }
 `;

export const PATIENT = gql`
    query(
        $id: token!
    )
    {
        Patient(_id: $id) {
            name {
                text
            }
            gender
        }
    }
 `;
