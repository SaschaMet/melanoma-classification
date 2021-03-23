import { gql } from '@apollo/client';

export const MEDIA = gql`
    query ($id: token) {
        Media(_id: $id) {
            id
            status
            identifier {
            value
            }
            content {
            url
            }
            note {
            text
            }
            bodySite {
            text
            }
        }
    }
 `;

export const MEDIA_BY_ENCOUNTER_ID = gql`
    query ($id: token) {
        MediaByEncounterId(_id: $id) {
            entries
        }
    }
 `;
