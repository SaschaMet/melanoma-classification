const { GraphQLString } = require('graphql');
const DateScalar = require('../scalars/date.scalar.js');
const TokenScalar = require('../scalars/token.scalar.js');

/**
 * @name exports
 * @static
 * @summary Arguments for the claimresponse query
 */
module.exports = {
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-created
	created: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'ClaimResponse.created',
		description: 'The creation date',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-disposition
	disposition: {
		type: GraphQLString,
		fhirtype: 'string',
		xpath: 'ClaimResponse.disposition',
		description: 'The contents of the disposition message',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-identifier
	identifier: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ClaimResponse.identifier',
		description: 'The identity of the ClaimResponse',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-insurer
	insurer: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'ClaimResponse.insurer',
		description: 'The organization which generated this resource',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-outcome
	outcome: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ClaimResponse.outcome',
		description: 'The processing outcome',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-patient
	patient: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'ClaimResponse.patient',
		description: 'The subject of care',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-payment-date
	payment_date: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'ClaimResponse.payment.date',
		description: 'The expected payment date',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-request
	request: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'ClaimResponse.request',
		description: 'The claim reference',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-requestor
	requestor: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'ClaimResponse.requestor',
		description: 'The Provider of the claim',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-status
	status: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ClaimResponse.status',
		description: 'The status of the ClaimResponse',
	},
	// http://hl7.org/fhir/SearchParameter/ClaimResponse-use
	use: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ClaimResponse.use',
		description: 'The type of claim',
	},
};
