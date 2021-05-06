const { GraphQLString } = require('graphql');
const TokenScalar = require('../scalars/token.scalar.js');
const UriScalar = require('../scalars/uri.scalar.js');
const DateScalar = require('../scalars/date.scalar.js');

/**
 * @name exports
 * @static
 * @summary Arguments for the communication query
 */
module.exports = {
	// http://hl7.org/fhir/SearchParameter/Communication-based-on
	based_on: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.basedOn',
		description: 'Request fulfilled by this communication',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-category
	category: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'Communication.category',
		description: 'Message category',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-encounter
	encounter: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.encounter',
		description: 'Encounter created as part of',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-identifier
	identifier: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'Communication.identifier',
		description: 'Unique identifier',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-instantiates-canonical
	instantiates_canonical: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.instantiatesCanonical',
		description: 'Instantiates FHIR protocol or definition',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-instantiates-uri
	instantiates_uri: {
		type: UriScalar,
		fhirtype: 'uri',
		xpath: 'Communication.instantiatesUri',
		description: 'Instantiates external protocol or definition',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-medium
	medium: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'Communication.medium',
		description: 'A channel of communication',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-part-of
	part_of: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.partOf',
		description: 'Part of this action',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-patient
	patient: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.subject',
		description: 'Focus of message',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-received
	received: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'Communication.received',
		description: 'When received',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-recipient
	recipient: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.recipient',
		description: 'Message recipient',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-sender
	sender: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.sender',
		description: 'Message sender',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-sent
	sent: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'Communication.sent',
		description: 'When sent',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-status
	status: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'Communication.status',
		description:
			'preparation | in-progress | not-done | suspended | aborted | completed | entered-in-error',
	},
	// http://hl7.org/fhir/SearchParameter/Communication-subject
	subject: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'Communication.subject',
		description: 'Focus of message',
	},
};
